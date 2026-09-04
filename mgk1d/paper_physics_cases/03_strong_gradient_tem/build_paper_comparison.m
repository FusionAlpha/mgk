function comparison = build_paper_comparison()
% Build the strong-gradient kinetic-electron mode comparison.

caseRoot = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(fileparts(caseRoot)));
mgkFile = fullfile(projectRoot, 'analysis', 'tem_eta_multiroot_ky07', ...
    ['tem_eta_multiroot_ky07_th8pi_nb24_eps0p018_etai0_eta3p13_' ...
    'open_extrapolated_direct10.mat']);
cgyroDirectory = fullfile(projectRoot, 'analysis', ...
    'cgyro_tem_eps0p018_etai0_etae3p13_ky07_nradial17_scaled_restart');

mgk = load(mgkFile, 'summary', 'thetaModes', 'phiModes');
thetaMgk = mgk.thetaModes{1}{1}(:);
phiMgk = mgk.phiModes{1}{1}(:);
omegaMgk = complex(mgk.summary.omega(1), mgk.summary.gamma(1));
[thetaCgyro, phiCgyro, omegaCgyro, nRadial] = ...
    readCgyroMode(cgyroDirectory);
assert(nRadial == 17, 'The CGYRO reference must use N_RADIAL=17.');

[phiCgyro, phiMgk, overlap] = alignModes( ...
    thetaCgyro, phiCgyro, thetaMgk, phiMgk);
relativeFrequencyDifference = abs(omegaMgk - omegaCgyro) / ...
    abs(omegaCgyro);
comparison = table(real(omegaMgk), imag(omegaMgk), ...
    real(omegaCgyro), imag(omegaCgyro), ...
    relativeFrequencyDifference, overlap, ...
    'VariableNames', {'mgkOmega', 'mgkGamma', ...
    'cgyroOmega', 'cgyroGamma', ...
    'relativeFrequencyDifference', 'modeOverlap'});

writetable(comparison, fullfile(caseRoot, ...
    'salpha_strong_gradient_tem_mode_cgyro_mgk.csv'));
save(fullfile(caseRoot, ...
    'salpha_strong_gradient_tem_mode_cgyro_mgk.mat'), ...
    'comparison', 'omegaMgk', 'omegaCgyro', 'thetaMgk', 'phiMgk', ...
    'thetaCgyro', 'phiCgyro');

figureDirectory = fullfile(fileparts(projectRoot), 'paper', 'figures');
if ~isfolder(figureDirectory)
    mkdir(figureDirectory);
end
pdfFile = fullfile(figureDirectory, ...
    'salpha_strong_gradient_tem_mode_cgyro_mgk.pdf');
pngFile = fullfile(figureDirectory, ...
    'salpha_strong_gradient_tem_mode_cgyro_mgk.png');
plotComparison(thetaMgk, phiMgk, thetaCgyro, phiCgyro, ...
    pdfFile, pngFile);

fprintf('Strong-gradient kinetic-electron mode comparison\n');
fprintf('  MGK:   %+.8f%+.8fi\n', real(omegaMgk), imag(omegaMgk));
fprintf('  CGYRO: %+.8f%+.8fi\n', real(omegaCgyro), imag(omegaCgyro));
fprintf('  relative complex-frequency difference: %.5f%%\n', ...
    100 * relativeFrequencyDifference);
fprintf('  phase-aligned overlap: %.7f\n', overlap);
fprintf('  figure: %s\n', pdfFile);
disp(comparison)
end

function [thetaBalloon, phi, omega, nRadial] = readCgyroMode(directory)
infoFile = fullfile(directory, 'out.cgyro.info');
assert(isfile(infoFile) && contains(fileread(infoFile), ...
    'Linear converged'), 'The CGYRO reference is not converged.');

grid = sscanf(fileread(fullfile(directory, 'out.cgyro.grids')), '%f');
nRadial = round(grid(4));
nTheta = round(grid(5));
radialHarmonic = grid(12:(11 + nRadial));
theta = grid((12 + nRadial):(11 + nRadial + nTheta));

fileId = fopen(fullfile(directory, 'bin.cgyro.phib'), 'rb', 'ieee-le');
assert(fileId >= 0, 'Cannot open the CGYRO potential output.');
cleanup = onCleanup(@() fclose(fileId));
raw = fread(fileId, inf, 'single=>double');
clear cleanup;
frameSize = 2 * nTheta * nRadial;
assert(~isempty(raw) && mod(numel(raw), frameSize) == 0, ...
    'The CGYRO potential history is incomplete.');
history = reshape(complex(raw(1:2:end), raw(2:2:end)), ...
    nTheta, nRadial, []);
finalMode = history(:, :, end);
assert(all(isfinite(finalMode), 'all'), ...
    'The final CGYRO potential frame is not finite.');
thetaBalloon = theta + 2 * pi * radialHarmonic.';
[thetaBalloon, order] = sort(thetaBalloon(:));
phi = finalMode(order);

frequency = sscanf(fileread(fullfile(directory, 'out.cgyro.freq')), '%f');
frequency = reshape(frequency, 2, []).';
majorToMinorRatio = 2.7777777777777777;
omega = majorToMinorRatio * complex(frequency(end, 1), frequency(end, 2));
end

function [cgyro, mgk, overlap] = alignModes( ...
        cgyroTheta, cgyro, mgkTheta, mgk)
cgyro = normalizeAtPeak(cgyro);
mgk = normalizeAtPeak(mgk);
inside = cgyroTheta >= min(mgkTheta) & cgyroTheta <= max(mgkTheta);
sampledMgk = interp1(mgkTheta, mgk, cgyroTheta(inside), 'pchip');
sampledCgyro = cgyro(inside);
phaseProduct = sum(conj(sampledMgk) .* sampledCgyro);
assert(abs(phaseProduct) > eps, 'Mode phase alignment is ill-conditioned.');
phase = phaseProduct / abs(phaseProduct);
mgk = phase * mgk;
sampledMgk = phase * sampledMgk;
overlap = abs(sampledMgk' * sampledCgyro) / ...
    (norm(sampledMgk) * norm(sampledCgyro));
end

function mode = normalizeAtPeak(mode)
mode = mode / max(abs(mode));
[~, peak] = max(abs(mode));
mode = mode * exp(-1i * angle(mode(peak)));
end

function plotComparison(thetaMgk, phiMgk, thetaCgyro, phiCgyro, ...
        pdfFile, pngFile)
mgkColor = [0.00, 0.45, 0.70];
cgyroColor = [0.90, 0.62, 0.00];
neutral = [0.16, 0.16, 0.16];
figureHandle = figure('Visible', 'off', 'Color', 'white', ...
    'Units', 'inches', 'Position', [0.5, 0.5, 3.45, 2.55], ...
    'Renderer', 'painters');
axisHandle = axes(figureHandle);
hold(axisHandle, 'on');

mgkMarkers = markerIndices(thetaMgk, 13);
cgyroMarkers = markerIndices(thetaCgyro, 15);
plot(axisHandle, thetaMgk / pi, real(phiMgk), '-o', ...
    'Color', mgkColor, 'LineWidth', 1.15, 'MarkerSize', 2.5, ...
    'MarkerIndices', mgkMarkers, 'MarkerFaceColor', 'white', ...
    'HandleVisibility', 'off');
plot(axisHandle, thetaMgk / pi, imag(phiMgk), '--o', ...
    'Color', mgkColor, 'LineWidth', 1.05, 'MarkerSize', 2.5, ...
    'MarkerIndices', mgkMarkers, 'MarkerFaceColor', 'white', ...
    'HandleVisibility', 'off');
plot(axisHandle, thetaCgyro / pi, real(phiCgyro), '-s', ...
    'Color', cgyroColor, 'LineWidth', 1.05, 'MarkerSize', 2.4, ...
    'MarkerIndices', cgyroMarkers, 'MarkerFaceColor', 'white', ...
    'HandleVisibility', 'off');
plot(axisHandle, thetaCgyro / pi, imag(phiCgyro), '--s', ...
    'Color', cgyroColor, 'LineWidth', 1.0, 'MarkerSize', 2.4, ...
    'MarkerIndices', cgyroMarkers, 'MarkerFaceColor', 'white', ...
    'HandleVisibility', 'off');
plot(axisHandle, [-8, 8], [0, 0], ':', 'Color', [0.60, 0.60, 0.60], ...
    'LineWidth', 0.55, 'HandleVisibility', 'off');

hReal = plot(axisHandle, NaN, NaN, '-', 'Color', neutral, ...
    'LineWidth', 1.1, 'DisplayName', 'Re $\phi$');
hImag = plot(axisHandle, NaN, NaN, '--', 'Color', neutral, ...
    'LineWidth', 1.1, 'DisplayName', 'Im $\phi$');
hMgk = plot(axisHandle, NaN, NaN, '-o', 'Color', mgkColor, ...
    'LineWidth', 1.1, 'MarkerSize', 3.0, 'MarkerFaceColor', 'white', ...
    'DisplayName', 'MGK');
hCgyro = plot(axisHandle, NaN, NaN, '-s', 'Color', cgyroColor, ...
    'LineWidth', 1.0, 'MarkerSize', 2.9, 'MarkerFaceColor', 'white', ...
    'DisplayName', 'CGYRO');

xlabel(axisHandle, '$\theta/\pi$', 'Interpreter', 'latex');
ylabel(axisHandle, '$\phi/\max|\phi|$', 'Interpreter', 'latex');
xlim(axisHandle, [-8, 8]);
ylim(axisHandle, [-1.05, 1.05]);
axisHandle.XTick = -8:4:8;
axisHandle.YTick = [-1, -0.5, 0, 0.5, 1];
grid(axisHandle, 'on');
box(axisHandle, 'on');

legendHandle = legend(axisHandle, [hReal, hImag, hMgk, hCgyro], ...
    'Location', 'northwest', 'NumColumns', 2, 'Interpreter', 'latex');
legendHandle.Box = 'off';
legendHandle.FontSize = 6.5;

axisHandle.FontName = 'Times New Roman';
axisHandle.FontSize = 7.3;
axisHandle.LineWidth = 0.6;
axisHandle.TickDir = 'out';
axisHandle.GridAlpha = 0.12;
axisHandle.GridColor = [0.55, 0.55, 0.55];
axisHandle.Position = [0.15, 0.15, 0.82, 0.81];

exportgraphics(figureHandle, pdfFile, 'ContentType', 'vector', ...
    'BackgroundColor', 'white');
exportgraphics(figureHandle, pngFile, 'Resolution', 300, ...
    'BackgroundColor', 'white');
close(figureHandle);
end

function indices = markerIndices(theta, count)
inside = find(theta >= -8 * pi & theta <= 8 * pi);
indices = unique(round(linspace(inside(1), inside(end), count)));
end
