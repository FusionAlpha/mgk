function summary = build_paper_comparison()
% Build the Miller TEM ballooning-domain comparison for Section 4.3.

caseRoot = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(fileparts(caseRoot)));
case5 = fullfile(projectRoot, 'analysis', ...
    'cgyro_miller_rewoldt2007_tem');
case9 = fullfile(projectRoot, 'analysis', ...
    'cgyro_miller_rewoldt2007_tem_nradial9');
checkpointFile = fullfile(case5, 'mgk_checkpoint.mat');

[theta5, phi5, omega5, nRadial5] = readCgyroCase(case5);
[theta9, phi9, omega9, nRadial9] = readCgyroCase(case9);
assert(nRadial5 == 5 && nRadial9 == 9, ...
    'The CGYRO cases do not have the expected radial extents.');

checkpoint = load(checkpointFile, 'mgk', 'mgkOmega');
assert(isfield(checkpoint, 'mgk') && isfield(checkpoint, 'mgkOmega'), ...
    'The MGK checkpoint is missing the stored mode or frequency.');
thetaMgk = checkpoint.mgk.theta(:);
phiMgk = checkpoint.mgk.phi(:);
omegaMgk = checkpoint.mgkOmega;
assert(numel(thetaMgk) == numel(phiMgk) && ...
    all(isfinite(thetaMgk)) && all(isfinite(phiMgk)), ...
    'The stored MGK mode is invalid.');

phi9 = normalizeAtPeak(phi9);
phi5 = alignToReference(theta9, phi9, theta5, normalizeAtPeak(phi5));
phiMgk = alignToReference(theta9, phi9, thetaMgk, ...
    normalizeAtPeak(phiMgk));

overlap5to9 = modeOverlap(theta9, phi9, theta5, phi5);
overlapMgkTo9 = modeOverlap(theta9, phi9, thetaMgk, phiMgk);
edgeThreshold5 = outerTurnThreshold(theta5);
edgeThreshold9 = outerTurnThreshold(theta9);
edgeThresholdMgk = outerTurnThreshold(thetaMgk);
edge5 = max(abs(phi5(abs(theta5) >= edgeThreshold5)));
edge9 = max(abs(phi9(abs(theta9) >= edgeThreshold9)));
edgeMgk = max(abs(phiMgk(abs(thetaMgk) >= edgeThresholdMgk)));

model = ["CGYRO N_r=5"; "CGYRO N_r=9"; "MGK open"];
omega = [omega5; omega9; omegaMgk];
thetaMinPi = [min(theta5); min(theta9); min(thetaMgk)] / pi;
thetaMaxPi = [max(theta5); max(theta9); max(thetaMgk)] / pi;
edgeThresholdPi = [edgeThreshold5; edgeThreshold9; ...
    edgeThresholdMgk] / pi;
outerTurnMax = [edge5; edge9; edgeMgk];
overlapWithN9 = [overlap5to9; 1; overlapMgkTo9];
summary = table(model, real(omega), imag(omega), thetaMinPi, ...
    thetaMaxPi, edgeThresholdPi, outerTurnMax, overlapWithN9, ...
    'VariableNames', {'model', 'omegaAOverCs', 'gammaAOverCs', ...
    'thetaMinPi', 'thetaMaxPi', 'edgeThresholdPi', ...
    'outerTurnMax', 'overlapWithN9'});

csvFile = fullfile(caseRoot, 'miller_tem_domain_cgyro_mgk.csv');
matFile = fullfile(caseRoot, 'miller_tem_domain_cgyro_mgk.mat');
writetable(summary, csvFile);
save(matFile, 'summary', 'omega5', 'omega9', 'omegaMgk', ...
    'theta5', 'theta9', 'thetaMgk', 'phi5', 'phi9', 'phiMgk', ...
    'overlap5to9', 'overlapMgkTo9', 'edge5', 'edge9', 'edgeMgk');

figureDirectory = fullfile(fileparts(projectRoot), 'paper', 'figures');
if ~isfolder(figureDirectory)
    mkdir(figureDirectory);
end
figureFile = fullfile(figureDirectory, ...
    'miller_tem_domain_cgyro_mgk.pdf');
plotComparison(theta5, phi5, theta9, phi9, thetaMgk, phiMgk, ...
    figureFile);

fprintf('Miller TEM ballooning-domain comparison\n');
fprintf('  CGYRO N_r=5 to N_r=9 overlap: %.6f\n', overlap5to9);
fprintf('  MGK to CGYRO N_r=9 overlap: %.6f\n', overlapMgkTo9);
fprintf('  outer-turn amplitudes: %.4g, %.4g, %.4g\n', ...
    edge5, edge9, edgeMgk);
fprintf('  data:   %s\n', csvFile);
fprintf('  figure: %s\n', figureFile);
disp(summary)
end

function [thetaBalloon, phi, omega, nRadial] = readCgyroCase(directory)
infoFile = fullfile(directory, 'out.cgyro.info');
assert(isfile(infoFile) && contains(fileread(infoFile), ...
    'Linear converged'), 'CGYRO case is not converged: %s', directory);

gridFile = fullfile(directory, 'out.cgyro.grids');
gridValues = sscanf(fileread(gridFile), '%f');
assert(numel(gridValues) >= 11, ...
    'Invalid CGYRO grid file: %s', directory);
nRadial = round(gridValues(4));
nTheta = round(gridValues(5));
assert(nRadial > 0 && nTheta > 0 && ...
    numel(gridValues) >= 11 + nRadial + nTheta, ...
    'Incomplete CGYRO grid file: %s', directory);
radialHarmonic = gridValues(12:(11 + nRadial));
theta = gridValues((12 + nRadial):(11 + nRadial + nTheta));

potentialFile = fullfile(directory, 'bin.cgyro.phib');
fileId = fopen(potentialFile, 'rb', 'ieee-le');
assert(fileId >= 0, 'Cannot open CGYRO potential output: %s', directory);
cleanup = onCleanup(@() fclose(fileId));
raw = fread(fileId, inf, 'single=>double');
clear cleanup;
frameSize = 2 * nTheta * nRadial;
assert(~isempty(raw) && mod(numel(raw), frameSize) == 0, ...
    'Invalid CGYRO potential history: %s', directory);
history = reshape(complex(raw(1:2:end), raw(2:2:end)), ...
    nTheta, nRadial, []);
finalFrame = history(:, :, end);
thetaBalloon = theta + 2 * pi * radialHarmonic.';
[thetaBalloon, order] = sort(thetaBalloon(:));
phi = finalFrame(order);

frequencyValues = sscanf(fileread(fullfile(directory, ...
    'out.cgyro.freq')), '%f');
assert(~isempty(frequencyValues) && mod(numel(frequencyValues), 2) == 0, ...
    'Invalid CGYRO frequency history: %s', directory);
frequencyHistory = reshape(frequencyValues, 2, []).';
omega = complex(frequencyHistory(end, 1), frequencyHistory(end, 2));
end

function mode = normalizeAtPeak(mode)
scale = max(abs(mode));
assert(scale > 0, 'Cannot normalize a zero mode.');
mode = mode / scale;
[~, peak] = max(abs(mode));
mode = mode * exp(-1i * angle(mode(peak)));
end

function mode = alignToReference(referenceTheta, referenceMode, theta, mode)
inside = referenceTheta >= min(theta) & referenceTheta <= max(theta);
assert(nnz(inside) >= 8, 'Insufficient common field-line interval.');
sample = interp1(theta, mode, referenceTheta(inside), 'pchip');
phaseProduct = sum(conj(sample) .* referenceMode(inside));
assert(abs(phaseProduct) > eps, 'Mode phase alignment is ill-conditioned.');
mode = mode * phaseProduct / abs(phaseProduct);
end

function value = modeOverlap(referenceTheta, referenceMode, theta, mode)
inside = referenceTheta >= min(theta) & referenceTheta <= max(theta);
sample = interp1(theta, mode, referenceTheta(inside), 'pchip');
reference = referenceMode(inside);
value = abs(sample' * reference) / (norm(sample) * norm(reference));
end

function threshold = outerTurnThreshold(theta)
threshold = max(abs(theta)) - pi;
assert(threshold >= 0 && any(abs(theta) >= threshold), ...
    'The mode does not contain an outer poloidal turn.');
end

function plotComparison(theta5, phi5, theta9, phi9, ...
        thetaMgk, phiMgk, outputFile)
mgkColor = [0.00, 0.45, 0.70];
n9Color = [0.90, 0.62, 0.00];
n5Color = [0.00, 0.62, 0.45];
dark = [0.16, 0.16, 0.16];
figureHandle = figure('Visible', 'off', 'Color', 'white', ...
    'Units', 'inches', 'Position', [0.5, 0.5, 7.15, 2.75]);
layout = tiledlayout(figureHandle, 1, 2, ...
    'TileSpacing', 'compact', 'Padding', 'compact');

axisEnvelope = nexttile(layout, 1);
hold(axisEnvelope, 'on');
mgkMarkers = markerIndices(thetaMgk, 17);
n9Markers = markerIndices(theta9, 19);
n5Markers = markerIndices(theta5, 15);
semilogy(axisEnvelope, thetaMgk / pi, max(abs(phiMgk), 1e-5), ...
    '-o', 'Color', mgkColor, 'LineWidth', 1.15, 'MarkerSize', 2.4, ...
    'MarkerIndices', mgkMarkers, 'MarkerFaceColor', 'white', ...
    'HandleVisibility', 'off');
semilogy(axisEnvelope, theta9 / pi, max(abs(phi9), 1e-5), ...
    '-s', 'Color', n9Color, 'LineWidth', 1.0, 'MarkerSize', 2.3, ...
    'MarkerIndices', n9Markers, 'MarkerFaceColor', 'white', ...
    'HandleVisibility', 'off');
semilogy(axisEnvelope, theta5 / pi, max(abs(phi5), 1e-5), ...
    '-^', 'Color', n5Color, 'LineWidth', 1.0, 'MarkerSize', 2.4, ...
    'MarkerIndices', n5Markers, 'MarkerFaceColor', 'white', ...
    'HandleVisibility', 'off');
axisEnvelope.YScale = 'log';
xlabel(axisEnvelope, '$\theta/\pi$', 'Interpreter', 'latex');
ylabel(axisEnvelope, '$|\phi|/\max|\phi|$', 'Interpreter', 'latex');
title(axisEnvelope, '(a)', 'FontWeight', 'normal');
xlim(axisEnvelope, [-9.2, 9.2]);
ylim(axisEnvelope, [1e-3, 1.15]);
axisEnvelope.XTick = -8:2:8;
grid(axisEnvelope, 'on');
box(axisEnvelope, 'on');
formatAxis(axisEnvelope);

axisMode = nexttile(layout, 2);
hold(axisMode, 'on');
plot(axisMode, thetaMgk / pi, real(phiMgk), '-o', ...
    'Color', mgkColor, 'LineWidth', 1.15, 'MarkerSize', 2.4, ...
    'MarkerIndices', mgkMarkers, 'MarkerFaceColor', 'white', ...
    'HandleVisibility', 'off');
plot(axisMode, theta9 / pi, real(phi9), '-s', ...
    'Color', n9Color, 'LineWidth', 1.0, 'MarkerSize', 2.3, ...
    'MarkerIndices', n9Markers, 'MarkerFaceColor', 'white', ...
    'HandleVisibility', 'off');
plot(axisMode, thetaMgk / pi, imag(phiMgk), '--o', ...
    'Color', mgkColor, 'LineWidth', 1.15, 'MarkerSize', 2.4, ...
    'MarkerIndices', mgkMarkers, 'MarkerFaceColor', 'white', ...
    'HandleVisibility', 'off');
plot(axisMode, theta9 / pi, imag(phi9), '--s', ...
    'Color', n9Color, 'LineWidth', 1.0, 'MarkerSize', 2.3, ...
    'MarkerIndices', n9Markers, 'MarkerFaceColor', 'white', ...
    'HandleVisibility', 'off');
yline(axisMode, 0, ':', 'Color', [0.60, 0.60, 0.60], ...
    'LineWidth', 0.55, 'HandleVisibility', 'off');
xlabel(axisMode, '$\theta/\pi$', 'Interpreter', 'latex');
ylabel(axisMode, '$\phi/\max|\phi|$', 'Interpreter', 'latex');
title(axisMode, '(b)', 'FontWeight', 'normal');
xlim(axisMode, [-5, 5]);
ylim(axisMode, [-1.05, 1.05]);
axisMode.XTick = -4:2:4;
axisMode.YTick = [-1, -0.5, 0, 0.5, 1];
grid(axisMode, 'on');
box(axisMode, 'on');
formatAxis(axisMode);

hReal = plot(axisMode, NaN, NaN, '-', 'Color', dark, ...
    'LineWidth', 1.15, 'DisplayName', 'Re $\phi$');
hImag = plot(axisMode, NaN, NaN, '--', 'Color', dark, ...
    'LineWidth', 1.15, 'DisplayName', 'Im $\phi$');
hMgk = plot(axisMode, NaN, NaN, '-o', 'Color', mgkColor, ...
    'LineWidth', 1.15, 'MarkerSize', 3.1, ...
    'MarkerFaceColor', 'white', 'DisplayName', 'MGK open');
hN9 = plot(axisMode, NaN, NaN, '-s', 'Color', n9Color, ...
    'LineWidth', 1.0, 'MarkerSize', 3.0, ...
    'MarkerFaceColor', 'white', 'DisplayName', 'CGYRO $N_r=9$');
hN5 = plot(axisMode, NaN, NaN, '-^', 'Color', n5Color, ...
    'LineWidth', 1.0, 'MarkerSize', 3.1, ...
    'MarkerFaceColor', 'white', 'DisplayName', 'CGYRO $N_r=5$');
legendHandle = legend(axisMode, [hReal, hImag, hMgk, hN9, hN5], ...
    'Location', 'southoutside', 'Orientation', 'horizontal', ...
    'NumColumns', 5, 'Interpreter', 'latex');
legendHandle.Layout.Tile = 'south';
legendHandle.Box = 'off';
legendHandle.FontSize = 7.0;

exportgraphics(figureHandle, outputFile, 'ContentType', 'vector', ...
    'BackgroundColor', 'white');
close(figureHandle);
end

function indices = markerIndices(theta, count)
inside = find(theta >= -5 * pi & theta <= 5 * pi);
if isempty(inside)
    inside = (1:numel(theta)).';
end
indices = unique(round(linspace(inside(1), inside(end), count)));
end

function formatAxis(axisHandle)
axisHandle.FontName = 'Times New Roman';
axisHandle.FontSize = 7.3;
axisHandle.LineWidth = 0.6;
axisHandle.TickDir = 'out';
axisHandle.GridAlpha = 0.12;
axisHandle.MinorGridAlpha = 0.08;
end
