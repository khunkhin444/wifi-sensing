function csi_preprocess_window_export_v2(varargin)
% CSI Preprocess & Windowing (2.4 GHz default) with Phase Export
% - Parses PicoScenes/PMT .csi via parseCSIFile.m
% - Time windows (WinSec/HopSec), drop by MinFillRatio
% - Per-window amplitude: Hampel -> SG/Wavelet -> clip -> trim/subsample
% - Phase pipeline added:
%   * per-packet: unwrap(k) -> linear slope remove (CSD/SFO) -> CPE removal
%   * optional: temporal unwrap over time + gentle high-pass
% - Exports:
%     X_window_%05d.npy         (amplitude, float32)
%     Xphase_window_%05d.npy    (phase, float32, radians after sanitization)
%     meta_%05d.mat
%
% REQUIRE: parseCSIFile.m in path.

%% ---------------- Params ----------------
p = inputParser;
addParameter(p,'OutDir','', @ischar);                 % default: <folder>/<basename>
addParameter(p,'UseCF_CBW_Filter',true, @islogical);  % try CF≈TargetCF & CBW=TargetCBW
addParameter(p,'TargetCF',5180, @isscalar);           % MHz (e.g., Ch10)
addParameter(p,'CFTol',15, @isscalar);                % MHz tolerance
addParameter(p,'TargetCBW',20, @isscalar);            % MHz band

% Time-based windowing
addParameter(p,'WinSec',3, @isscalar);
addParameter(p,'HopSec',0.5, @isscalar);
addParameter(p,'MinFillRatio',0.60, @isscalar);

% Sampling assumptions if timestamp missing/unstable
addParameter(p,'AssumedPktRate',800, @isscalar);      % pps

% Per-window preprocessing (amplitude)
addParameter(p,'HampelK',7, @isscalar);
addParameter(p,'DenoiseMethod','sgolay', @ischar);    % 'sgolay' or 'wavelet'
addParameter(p,'SG_Poly',3, @isscalar);
addParameter(p,'SG_FrameSec',0.42, @isscalar);
addParameter(p,'Wavelet','sym4', @ischar);
addParameter(p,'WaveletLevel',3, @isscalar);
addParameter(p,'ClipPercentile',99.5, @isscalar);

% Subcarrier handling
addParameter(p,'TrimEdges',[0 0], @(x)isnumeric(x)&&numel(x)==2);
addParameter(p,'SubsampleStep',1, @isscalar);

% Phase export controls
addParameter(p,'ExportPhase',true, @islogical);
addParameter(p,'DoTemporalUnwrap',true, @islogical);
addParameter(p,'PhaseHighpassHz',0.05, @isscalar);    % 0 disables HP

% Labels & meta
addParameter(p,'SessionID','', @ischar);
addParameter(p,'y_presence','', @ischar);             % 'empty'|'stationary'|''
addParameter(p,'y_loc','', @ischar);                  % RP id or '(x,y)'

% Verbosity
addParameter(p,'Verbose',true, @islogical);

parse(p, varargin{:});
opt = p.Results;

%% ---------------- File select ----------------
[filename, pathname] = uigetfile('*.csi', 'Select a CSI file');
if isequal(filename, 0), disp('User canceled.'); return; end
fullpath = fullfile(pathname, filename);
[~, base, ~] = fileparts(filename);

outDir = opt.OutDir;
if isempty(outDir), outDir = fullfile(pathname, base); end
if ~exist(outDir,'dir'), mkdir(outDir); end
winDir = fullfile(outDir, 'windows');
if ~exist(winDir,'dir'), mkdir(winDir); end

if isempty(opt.SessionID), opt.SessionID = base; end

fprintf('Start parsing PicoScenes CSI file: %s\n', filename);
S = parseCSIFile(fullpath);     % cell array of frames/segments
N = numel(S);
if opt.Verbose, fprintf('[seg] segments: %d\n', N); end

%% -------- Collect rows (Amplitude + Phase) with optional CF/CBW filtering ------
A_rows = {};  PHI_rows = {};  SCIDX_rows = {};
t_rows = [];  cf_rows = [];   bw_rows = [];  nsc_rows = [];

for k = 1:N
    Sk = S{k};
    if ~isstruct(Sk) || ~isfield(Sk,'CSI'), continue; end
    C = Sk.CSI;

    % ---- Amplitude source preference: Mag -> Amp -> |CSI|
    amp = [];
    if isfield(C,'Mag'), amp = double(C.Mag); end
    if isempty(amp) && isfield(C,'Amp'), amp = double(C.Amp); end
    if isempty(amp)
        z = [];
        if isfield(C,'CSI'), z = C.CSI; end
        if isempty(z) && isfield(C,'csi'), z = C.csi; end
        if ~isempty(z), amp = abs(double(squeeze(z))); end
    end
    if isempty(amp), continue; end
    amp = squeeze(amp);

    % ---- Phase source preference: Phase -> angle(CSI)
    phi = [];
    if isfield(C,'Phase') && ~isempty(C.Phase)
        phi = double(C.Phase);
    else
        z = [];
        if isfield(C,'CSI'), z = C.CSI; end
        if isempty(z) && isfield(C,'csi'), z = C.csi; end
        if ~isempty(z), phi = angle(double(squeeze(z))); end
    end
    % If phase totally missing, we still export amplitude. We'll fill PHI later if enabled.

    % 3D -> collapse non-subcarrier dims into rows
    [amp, Nsc_guess] = ensure_rows_last_is_subc(amp);
    if ~isempty(phi)
        [phi, ~] = ensure_rows_last_is_subc(phi);
    end

    % If shapes disagree (rare), try to coerce phi to same Nsc dimension
    if ~isempty(phi) && size(phi,2) ~= size(amp,2)
        min_n = min(size(phi,2), size(amp,2));
        amp = amp(:,1:min_n);
        phi = phi(:,1:min_n);
        Nsc_guess = min_n;
    end

    % Subcarrier index (if available)
    scidx = [];
    if isfield(C,'SubcarrierIndex') && ~isempty(C.SubcarrierIndex)
        v = double(C.SubcarrierIndex);
        if size(v,1) == 1 && Nsc_guess == numel(v)
            scidx = repmat(v, size(amp,1), 1);
        elseif size(v,2) == Nsc_guess
            scidx = v;
        end
    end

    % Split bundled blocks into per-row entries
    Tk = size(amp,1);
    for i=1:Tk
        A_rows{end+1,1} = amp(i,:); %#ok<AGROW>
        if ~isempty(phi)
            PHI_rows{end+1,1} = phi(i,:); %#ok<AGROW>
        else
            PHI_rows{end+1,1} = []; %#ok<AGROW>
        end
        if ~isempty(scidx)
            SCIDX_rows{end+1,1} = scidx(i,:); %#ok<AGROW>
        else
            SCIDX_rows{end+1,1} = []; %#ok<AGROW>
        end

        t_rows(end+1,1) = get_ts_one(C, min(i, get_len_field(C,'Timestamp','TimingOffsets'))); %#ok<AGROW>
        [cfMHz, cbwMHz] = get_cf_cbw(C, i);
        cf_rows(end+1,1) = cfMHz; %#ok<AGROW>
        bw_rows(end+1,1) = cbwMHz; %#ok<AGROW>
        nsc_rows(end+1,1)= size(amp,2); %#ok<AGROW>
    end
end

assert(~isempty(A_rows), 'No CSI amplitude rows found.');

% ---- CF/CBW filter with safe fallback
keep = true(size(A_rows));
if opt.UseCF_CBW_Filter
    cf_ok = isnan(cf_rows) | abs(cf_rows - opt.TargetCF) <= opt.CFTol;
    bw_ok = isnan(bw_rows) | (bw_rows == opt.TargetCBW);
    keep = cf_ok & bw_ok;
    if ~any(keep)
        warning('No frames survived CF≈%d±%d & CBW=%d. Keeping CBW only.', ...
            opt.TargetCF, opt.CFTol, opt.TargetCBW);
        keep = (isnan(bw_rows) | (bw_rows == opt.TargetCBW));
    end
end

[A_rows, PHI_rows, SCIDX_rows, t_rows, cf_rows, bw_rows, nsc_rows] = ...
    apply_keep_mask(keep, A_rows, PHI_rows, SCIDX_rows, t_rows, cf_rows, bw_rows, nsc_rows);

% ---- Choose common Nsc by majority (mode); drop others for *both* amp & phase
target_nsc = mode(nsc_rows);
keep_nsc = (nsc_rows == target_nsc);
[A_rows, PHI_rows, SCIDX_rows, t_rows, cf_rows, bw_rows, nsc_rows] = ...
    apply_keep_mask(keep_nsc, A_rows, PHI_rows, SCIDX_rows, t_rows, cf_rows, bw_rows, nsc_rows);

T = numel(A_rows);
Nsc = target_nsc;
if opt.Verbose
    ucf = unique(round(cf_rows(~isnan(cf_rows))));
    ucbw = unique(bw_rows(~isnan(bw_rows)));
    fprintf('[nsc] chosen Nsc=%d | CF≈%s MHz | CBW=%s MHz | T=%d\n', ...
        Nsc, mat2str(ucf.'), mat2str(ucbw.'), T);
end

% ---- Build A [T x Nsc] and t
A = zeros(T, Nsc, 'double');
t2 = nan(T,1); cf2 = nan(T,1); bw2 = nan(T,1);
trimmed = 0; skipped = 0;
for i = 1:T
    rowA = A_rows{i};
    if numel(rowA) < Nsc
        skipped = skipped + 1; continue;
    elseif numel(rowA) > Nsc
        rowA = rowA(1:Nsc); trimmed = trimmed + 1;
    end
    A(i,:) = rowA;
    t2(i) = t_rows(i);
    cf2(i)= cf_rows(i);
    bw2(i)= bw_rows(i);
end
A = A(1:(T-skipped),:);   % in practice skipped is rare after mode gating
t = t2(1:size(A,1));
cf_rows = cf2(1:size(A,1));
bw_rows = bw2(1:size(A,1));
T = size(A,1);

% ---- Time & fs
if all(~isfinite(t))
    fs = opt.AssumedPktRate;
    t = (0:T-1)'/max(fs,1);
else
    t = fillmissing(t,'linear','EndValues','extrap');
    dt = median(diff(t));
    fs = (isfinite(dt)&&dt>0) * (1/dt) + (~(isfinite(dt)&&dt>0)) * opt.AssumedPktRate;
end
dur = t(end)-t(1);
pps = (T-1)/max(dur,eps);
if opt.Verbose
    fprintf('[*] CONCAT: %d pkts, %d subcarriers, fs=%.2f Hz, dur=%.1fs, pps=%.2f | trimmed=%d, skipped=%d\n', ...
        T,Nsc,fs,dur,pps,trimmed,skipped);
end

%% ---- Build Phase matrix PHI [T x Nsc] (sanitized)
if opt.ExportPhase
    PHI = zeros(T, Nsc, 'double');
    for i = 1:T
        rawPhi = PHI_rows{i};
        if isempty(rawPhi)
            % no phase for this packet -> mark NaN row
            PHI(i,:) = nan(1,Nsc);
            continue;
        end
        if numel(rawPhi) < Nsc, rawPhi(end+1:Nsc) = rawPhi(end); end
        if numel(rawPhi) > Nsc, rawPhi = rawPhi(1:Nsc); end

        scidx = SCIDX_rows{i};
        if isempty(scidx) || numel(scidx) ~= Nsc
            scidx = 1:Nsc;
        end
        PHI(i,:) = sanitize_phase_per_packet(rawPhi, scidx);
    end

    % Remove fully NaN rows (if any)
    good = any(isfinite(PHI),2);
    if ~all(good)
        A   = A(good,:);   t = t(good);
        PHI = PHI(good,:);
        T = size(A,1);
    end

    % Optional: temporal unwrap & gentle HP over time
    if opt.DoTemporalUnwrap || (opt.PhaseHighpassHz > 0)
        PHI = temporal_unwrap_and_hp(PHI, fs, opt.DoTemporalUnwrap, opt.PhaseHighpassHz);
    end
else
    PHI = []; % not used
end

%% ---------------- Window plan ----------------
win = max(4, round(opt.WinSec * fs));
hop = max(1, round(opt.HopSec * fs));
exp_frames = win;
min_frames = max(1, floor(opt.MinFillRatio * exp_frames));

starts = 1:hop:(T - win + 1);
stops  = starts + win - 1;
if opt.Verbose
    fprintf('[win] time: win=%d (~%.1fs), hop=%d (~%.1fs), min=%d (%.0f%%), candidates=%d\n', ...
        win, win/fs, hop, hop/fs, min_frames, opt.MinFillRatio*100, numel(starts));
end

%% ---------------- Subcarrier selection ----------------
L = max(0, min(opt.TrimEdges(1), Nsc-1));
R = max(0, min(opt.TrimEdges(2), Nsc-1-L));
if (1+L) > (Nsc-R)
    warning('TrimEdges too aggressive (Nsc=%d, L=%d, R=%d). Resetting to [0 0].', Nsc, L, R);
    L = 0; R = 0;
end
keep_idx = (1+L):(Nsc-R);
if opt.SubsampleStep > 1
    keep_idx = keep_idx(1:opt.SubsampleStep:end);
end
if isempty(keep_idx), keep_idx = 1:Nsc; end
keep_idx = keep_idx(:).';
S_kept = numel(keep_idx);
if opt.Verbose
    fprintf('[export] Nsc_raw=%d, Trim=[%d %d], Step=%d -> Nsc_kept=%d\n', Nsc, L, R, opt.SubsampleStep, S_kept);
end

%% ---------------- Per-window export ----------------
exported = 0;
for w = 1:numel(starts)
    idx = starts(w):stops(w);
    if numel(idx) < min_frames, continue; end

    % --- amplitude preprocessing per subcarrier (window-local) ---
    segA = A(idx, :);
    for c = 1:Nsc, segA(:,c) = hampel_safe(segA(:,c), opt.HampelK); end
    switch lower(opt.DenoiseMethod)
        case 'sgolay'
            frameLen = max(5, 2*floor( (opt.SG_FrameSec*fs)/2 ) + 1); % odd
            for c = 1:Nsc, segA(:,c) = sgolay_safely(segA(:,c), opt.SG_Poly, frameLen); end
        case 'wavelet'
            for c = 1:Nsc, segA(:,c) = wavelet_safely(segA(:,c), opt.Wavelet, opt.WaveletLevel); end
        otherwise
            error('Unknown DenoiseMethod: %s', opt.DenoiseMethod);
    end
    clipv = prctile(segA, opt.ClipPercentile, 1);
    segA = min(segA, repmat(clipv, size(segA,1), 1));
    Xw = segA(:, keep_idx);     % amplitude export

    % --- phase selection for this window (already sanitized globally) ---
    if opt.ExportPhase && ~isempty(PHI)
        segP = PHI(idx, :);
        Xpw  = segP(:, keep_idx);
    end

    % --- export ---
    exported = exported + 1;
    xfile = fullfile(winDir, sprintf('X_window_%05d.npy', exported));
    write_npy(xfile, single(Xw));

    if opt.ExportPhase && ~isempty(PHI)
        xpfile = fullfile(winDir, sprintf('Xphase_window_%05d.npy', exported));
        write_npy(xpfile, single(Xpw));
    end

    % meta
    CF = round(nanmedian(cf_rows));   % may be NaN
    CBW= nanmedian(bw_rows);          % may be NaN
    meta = struct();
    meta.session_id   = opt.SessionID;
    meta.window_index = exported;
    meta.fs           = fs;
    meta.t0           = t(idx(1)) - t(1);
    meta.WinSec       = opt.WinSec;
    meta.HopSec       = opt.HopSec;
    meta.MinFillRatio = opt.MinFillRatio;
    meta.CF_MHz       = CF;
    meta.CBW_MHz      = CBW;
    meta.Nsc_raw      = Nsc;
    meta.subcarrier_idx_kept = keep_idx;
    meta.Nsc_kept     = S_kept;
    meta.TrimEdges    = [L R];
    meta.SubsampleStep= opt.SubsampleStep;
    meta.ExportPhase  = opt.ExportPhase;
    if opt.ExportPhase
        meta.DoTemporalUnwrap = opt.DoTemporalUnwrap;
        meta.PhaseHighpassHz  = opt.PhaseHighpassHz;
        meta.phase_shape      = size(Xpw);
    end
    meta.y_presence   = opt.y_presence;
    meta.y_loc        = opt.y_loc;

    save(fullfile(winDir, sprintf('meta_%05d.mat', exported)), 'meta');

    if opt.Verbose && mod(exported, 200)==0
        fprintf('  exported %d windows...\n', exported);
    end
end

fprintf('[✓] Done. Exported %d window(s) to %s\n', exported, winDir);

end % ===== main =====

%% ================= Helpers =================
function [M, nsc] = ensure_rows_last_is_subc(M)
M = squeeze(M);
if ndims(M) == 3
    s = size(M); nsc = s(end);
    M = reshape(M, [], nsc);
else
    [r,c] = size(M);
    if c <= 4 && r > c     % heuristic: columns are streams, rows are tones
        M = M.';
        [~,c] = size(M);
    end
    nsc = c;
end
end

function [A_rows, PHI_rows, SCIDX_rows, t_rows, cf_rows, bw_rows, nsc_rows] = ...
    apply_keep_mask(keep, A_rows, PHI_rows, SCIDX_rows, t_rows, cf_rows, bw_rows, nsc_rows)
A_rows   = A_rows(keep);
PHI_rows = PHI_rows(keep);
SCIDX_rows = SCIDX_rows(keep);
t_rows   = t_rows(keep);
cf_rows  = cf_rows(keep);
bw_rows  = bw_rows(keep);
nsc_rows = nsc_rows(keep);
end

function ts = get_ts_one(C, i)
ts = NaN;
if isfield(C,'Timestamp')
    v = double(C.Timestamp);
    if ~isempty(v), ts = v(min(i,numel(v))); end
elseif isfield(C,'TimingOffsets')
    v = double(C.TimingOffsets);
    if ~isempty(v), ts = v(min(i,numel(v))); end
end
end

function L = get_len_field(C, f1, f2)
L = 0;
if isfield(C,f1), v = C.(f1); if ~isempty(v), L = max(L,numel(v)); end, end
if isfield(C,f2), v = C.(f2); if ~isempty(v), L = max(L,numel(v)); end, end
end

function [cfMHz, cbwMHz] = get_cf_cbw(C, i)
cfMHz = NaN; cbwMHz = NaN;
if isfield(C,'CarrierFreq')
    v = double(C.CarrierFreq); if ~isempty(v), cfMHz = pick_idx(v,i); end
elseif isfield(C,'cf')
    v = double(C.cf);          if ~isempty(v), cfMHz = pick_idx(v,i); end
end
if isfinite(cfMHz)
    if cfMHz > 1e6, cfMHz = cfMHz/1e6;
    elseif cfMHz > 1e3, cfMHz = cfMHz/1e3; end
end
if isfield(C,'CBW')
    v = double(C.CBW);     if ~isempty(v), cbwMHz = pick_idx(v,i); end
elseif isfield(C,'Pkt_CBW')
    v = double(C.Pkt_CBW); if ~isempty(v), cbwMHz = pick_idx(v,i); end
elseif isfield(C,'cbw')
    v = double(C.cbw);     if ~isempty(v), cbwMHz = pick_idx(v,i); end
end
end

function s = pick_idx(v,i)
v = v(:); s = v(min(i,numel(v)));
end

function x = hampel_safe(x, k)
if isempty(k) || k < 1, k = 5; end
try, x = hampel(x, k);
catch
    n = numel(x); y = x;
    for i = 1:n
        i1 = max(1,i-k); i2 = min(n,i+k);
        win = x(i1:i2); med = median(win);
        madv = median(abs(win-med)) + eps;
        if abs(x(i)-med) > 3*madv, y(i) = med; end
    end
    x = y;
end
end

function y = sgolay_safely(x, p, framelen)
try, y = sgolayfilt(x, p, framelen);
catch
    y = movmedian(x, max(3,round(framelen/4)));
    y = movmean(y,   max(3,round(framelen/4)));
end
end

function y = wavelet_safely(x, wname, lvl)
try, y = wdenoise(x, lvl, 'Wavelet', wname);
catch, y = sgolay_safely(x, 3, 11);
end
end

function ph = sanitize_phase_per_packet(phi_row, sc_idx)
% unwrap across subcarriers, remove linear slope, then remove CPE
ph = unwrap(phi_row);                   % across tones
x = double(sc_idx(:)); y = double(ph(:));
if numel(x) >= 3
    p = polyfit(x, y, 1);              % slope/intercept vs tone index
    y = y - polyval(p, x);
end
y = y - median(y,'omitnan');           % common phase error removal
ph = reshape(y, 1, []);
end

function PHI = temporal_unwrap_and_hp(PHI, fs, doUnwrap, hpHz)
if doUnwrap, PHI = unwrap(PHI, [], 1); end
if nargin >= 4 && hpHz > 0 && isfinite(fs) && fs > 0 && size(PHI,1) > 10
    try
        d = designfilt('highpassiir','FilterOrder',4, ...
            'HalfPowerFrequency', hpHz, 'SampleRate', fs);
        for c = 1:size(PHI,2), PHI(:,c) = filtfilt(d, PHI(:,c)); end
    catch
        PHI = detrend(PHI, 'linear');
    end
end
end

function write_npy(fname, A)
if islogical(A), A = uint8(A); end
switch class(A)
    case 'double', descr = '<f8';
    case 'single', descr = '<f4';
    case 'uint8',  descr = '|u1';
    case 'int8',   descr = '|i1';
    case 'uint16', descr = '<u2';
    case 'int16',  descr = '<i2';
    case 'uint32', descr = '<u4';
    case 'int32',  descr = '<i4';
    case 'uint64', descr = '<u8';
    case 'int64',  descr = '<i8';
    otherwise, error('write_npy: unsupported class %s', class(A));
end
sz = size(A);
shapeStr = sprintf('(%s)', strjoin(string(sz), ','));
headerDict = sprintf('{''descr'': ''%s'', ''fortran_order'': True, ''shape'': %s }', descr, shapeStr);
magic = uint8([147,'NUMPY']); ver = uint8([1 0]);
h = uint8(headerDict);
pad = 16 - mod(numel(magic)+2+2+numel(h), 16);
if pad == 0, pad = 16; end
h = [h, uint8(repmat(' ',1,pad-1)), uint8(sprintf('\n'))];
fid = fopen(fname,'w'); assert(fid>0, 'Cannot open %s', fname);
fwrite(fid, magic, 'uint8'); fwrite(fid, ver, 'uint8');
fwrite(fid, uint16(numel(h)), 'uint16', 0, 'ieee-le');
fwrite(fid, h, 'uint8'); fwrite(fid, A, class(A), 0, 'ieee-le'); fclose(fid);
end