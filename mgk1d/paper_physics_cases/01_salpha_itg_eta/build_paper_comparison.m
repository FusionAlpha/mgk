function comparison = build_paper_comparison()
% Build the matched MGK--CGYRO comparison used in Section 4.1.

caseRoot = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(fileparts(caseRoot)));
addpath(projectRoot);

etaI = (2.3:0.1:2.7).';
caseNames = {'eta_2p3', 'eta_2p4', 'eta_2p5', 'eta_2p6', 'eta_2p7'};
numCases = numel(etaI);

cgyroOmega = complex(NaN(numCases, 1));
thetaCgyro = cell(numCases, 1);
phiCgyroRaw = cell(numCases, 1);
for index = 1:numCases
    directory = fullfile(caseRoot, 'runs', 'ntheta_032', caseNames{index});
    [cgyroOmega(index), thetaCgyro{index}, phiCgyroRaw{index}] = ...
        readCgyroCase(directory);
end

baseConfig = matchedMgkConfig();
mgkResults = solveMgkContinuation(baseConfig, etaI);
mgkOmega = cellfun(@(result) result.omega, mgkResults);
thetaMgk = cellfun(@(result) result.theta(:), mgkResults, ...
    'UniformOutput', false);
phiMgkRaw = cellfun(@(result) result.phi(:), mgkResults, ...
    'UniformOutput', false);

phiCgyro = cell(numCases, 1);
phiMgk = cell(numCases, 1);
overlap = NaN(numCases, 1);
for index = 1:numCases
    [phiCgyro{index}, phiMgk{index}, overlap(index)] = alignModes( ...
        thetaCgyro{index}, phiCgyroRaw{index}, ...
        thetaMgk{index}, phiMgkRaw{index});
end

relativeComplexDifference = abs(mgkOmega - cgyroOmega) ./ abs(cgyroOmega);
continuationOverlap = adjacentModeOverlap(thetaMgk, phiMgkRaw);
assert(max(relativeComplexDifference) <= 0.02, ...
    'MGK--CGYRO complex-frequency difference exceeds 2%%.');
assert(min(overlap) >= 0.999, ...
    'MGK--CGYRO mode overlap is below 0.999.');

comparison = table(etaI, real(mgkOmega), imag(mgkOmega), ...
    real(cgyroOmega), imag(cgyroOmega), relativeComplexDifference, overlap, ...
    'VariableNames', {'etaI', 'mgkOmegaR', 'mgkGamma', ...
    'cgyroOmegaR', 'cgyroGamma', 'relativeComplexDifference', 'overlap'});
writetable(comparison, fullfile(caseRoot, ...
    'salpha_itg_eta_cgyro_mgk.csv'));
save(fullfile(caseRoot, 'salpha_itg_eta_cgyro_mgk.mat'), ...
    'comparison', 'etaI', 'mgkOmega', 'cgyroOmega', ...
    'thetaMgk', 'phiMgk', 'thetaCgyro', 'phiCgyro', ...
    'relativeComplexDifference', 'overlap', 'continuationOverlap', ...
    'baseConfig');

figureDirectory = fullfile(fileparts(projectRoot), 'paper', 'figures');
if ~isfolder(figureDirectory)
    mkdir(figureDirectory);
end
figureFile = fullfile(figureDirectory, ...
    'salpha_itg_eta_cgyro_mgk.pdf');
plotComparison(etaI, mgkOmega, cgyroOmega, thetaMgk, phiMgk, ...
    thetaCgyro, phiCgyro, figureFile);

fprintf('Matched adiabatic-electron ITG comparison\n');
fprintf('  maximum relative complex-frequency difference: %.4f%%\n', ...
    100 * max(relativeComplexDifference));
fprintf('  minimum phase-aligned mode overlap: %.7f\n', min(overlap));
fprintf('  minimum adjacent MGK mode overlap: %.7f\n', ...
    min(continuationOverlap));
fprintf('  figure: %s\n', figureFile);
end

function config = matchedMgkConfig()
elementaryCharge = 1.602176634e-19;
atomicMassUnit = 1.66053906660e-27;
magneticField = 2;
majorRadius = 4;
ionMass = 2 * atomicMassUnit;
ionTemperature = 1000;
ionThermalSpeed = sqrt(ionTemperature * elementaryCharge / ionMass);
ionCyclotronFrequency = elementaryCharge * magneticField / ionMass;
ionGyroradius = ionThermalSpeed / ionCyclotronFrequency;
frequencyReference = ionThermalSpeed / majorRadius;

config = struct();
config.physical = struct( ...
    'magneticField', magneticField, ...
    'majorRadius', majorRadius, ...
    'minorRadius', 0.2, ...
    'ionMass', ionMass, ...
    'ionChargeNumber', 1, ...
    'ionTemperature', ionTemperature, ...
    'electronTemperature', ionTemperature, ...
    'densityGradientLength', 1, ...
    'ionTemperatureGradientLength', 1 / 2.5, ...
    'binormalWavenumber', (0.45 / sqrt(2)) / ionGyroradius);
config.geometry = struct('model', 's-alpha', 'q', 1, ...
    'magneticShear', 1, 'alpha', 0, 'ballooningAngle', 0, ...
    'mirrorConvention', 'cgyro_s_alpha');
config.model = struct('magneticMirror', false, ...
    'electronClosure', 'adiabatic', 'aparallel', false, ...
    'bparallel', false, 'parallelBoundary', 'periodic');
config.grid = struct('thetaMin', -5 * pi, 'thetaMax', 5 * pi, ...
    'numTheta', 161, 'thetaMapAlpha', 3, 'energyMax', 12.5, ...
    'numEnergy', 16, 'numPitch', 16, 'numBouncePoints', 32);
config.solver = struct('useGpu', false, 'blockPrecision', 'double', ...
    'eigenBackend', 'eigs', 'modeSelection', 'nearest', ...
    'singleShiftTimeLimit', Inf, 'eigenTolerance', 1e-10, ...
    'eigenSubspaceDimension', 30, 'eigenMaxIterations', 1500, ...
    'eigenInitialVector', [], ...
    'frequencyGuess', (-0.79632 + 0.340784i) * frequencyReference, ...
    'enableFactorizationCache', true, ...
    'enableGpuResultCache', false, 'compactResult', false, ...
    'returnMatrices', false, 'computeResidual', true);
end

function results = solveMgkContinuation(baseConfig, etaI)
numCases = numel(etaI);
results = cell(numCases, 1);
center = find(abs(etaI - 2.5) < 10 * eps, 1);
results{center} = runPoint(baseConfig, etaI(center), []);

directions = {(center + 1):numCases, (center - 1):-1:1};
for direction = 1:numel(directions)
    previous = results{center};
    for index = directions{direction}
        results{index} = runPoint(baseConfig, etaI(index), previous);
        previous = results{index};
    end
end
end

function result = runPoint(baseConfig, etaI, previous)
config = baseConfig;
config.physical.ionTemperatureGradientLength = ...
    config.physical.densityGradientLength / etaI;
if ~isempty(previous)
    config.solver.frequencyGuess = ...
        previous.omega * previous.normalization.frequency;
    config.solver.eigenInitialVector = previous.reducedMode;
end
result = mgk1d.solve(config);
fprintf('MGK eta_i=%.1f omega=%+.9f%+.9fi residual=%.2e\n', ...
    etaI, real(result.omega), imag(result.omega), result.eigenResidual);
end

function [omega, thetaBalloon, phi] = readCgyroCase(directory)
infoFile = fullfile(directory, 'out.cgyro.info');
assert(isfile(infoFile) && contains(fileread(infoFile), 'Linear converged'), ...
    'CGYRO case is not converged: %s', directory);

frequencyValues = sscanf(fileread(fullfile(directory, ...
    'out.cgyro.freq')), '%f');
assert(~isempty(frequencyValues) && mod(numel(frequencyValues), 2) == 0, ...
    'Invalid CGYRO frequency history: %s', directory);
frequencyHistory = reshape(frequencyValues, 2, []).';
omega = 4 * complex(frequencyHistory(end, 1), frequencyHistory(end, 2));

gridValues = sscanf(fileread(fullfile(directory, ...
    'out.cgyro.grids')), '%f');
assert(numel(gridValues) >= 11, 'Invalid CGYRO grid file: %s', directory);
nRadial = round(gridValues(4));
nTheta = round(gridValues(5));
assert(nRadial > 0 && nTheta > 0 && ...
    numel(gridValues) >= 11 + nRadial + nTheta, ...
    'Incomplete CGYRO grid file: %s', directory);
radialHarmonic = gridValues(12:(11 + nRadial));
theta = gridValues((12 + nRadial):(11 + nRadial + nTheta));

fileId = fopen(fullfile(directory, 'bin.cgyro.phib'), 'rb', 'ieee-le');
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
end

function [cgyro, mgk, overlap] = alignModes(cgyroTheta, cgyro, mgkTheta, mgk)
cgyro = normalizeAtPeak(cgyro);
mgk = mgk / max(abs(mgk));
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

function overlap = adjacentModeOverlap(theta, modes)
numCases = numel(modes);
overlap = NaN(numCases - 1, 1);
for index = 1:(numCases - 1)
    first = interp1(theta{index}, modes{index}, theta{index + 1}, 'pchip');
    second = modes{index + 1};
    overlap(index) = abs(first' * second) / (norm(first) * norm(second));
end
end

function plotComparison(etaI, mgkOmega, cgyroOmega, thetaMgk, phiMgk, ...
        thetaCgyro, phiCgyro, outputFile)
mgkColor = [0.00, 0.45, 0.70];
cgyroColor = [0.90, 0.62, 0.00];
figureHandle = figure('Visible', 'off', 'Color', 'white', ...
    'Units', 'inches', 'Position', [0.5, 0.5, 7.15, 4.55], ...
    'Renderer', 'painters');
layout = tiledlayout(figureHandle, 2, 3, ...
    'TileSpacing', 'compact', 'Padding', 'compact');

axisHandle = nexttile(layout, 1);
hold(axisHandle, 'on');
plot(axisHandle, etaI, real(mgkOmega), '-o', 'Color', mgkColor, ...
    'LineWidth', 1.25, 'MarkerSize', 3.8, 'MarkerFaceColor', 'white', ...
    'DisplayName', 'MGK $\omega_r$');
plot(axisHandle, etaI, real(cgyroOmega), '-s', 'Color', cgyroColor, ...
    'LineWidth', 1.15, 'MarkerSize', 3.8, 'MarkerFaceColor', 'white', ...
    'DisplayName', 'CGYRO $\omega_r$');
plot(axisHandle, etaI, imag(mgkOmega), '--o', 'Color', mgkColor, ...
    'LineWidth', 1.25, 'MarkerSize', 3.8, 'MarkerFaceColor', 'white', ...
    'DisplayName', 'MGK $\gamma$');
plot(axisHandle, etaI, imag(cgyroOmega), '--s', 'Color', cgyroColor, ...
    'LineWidth', 1.15, 'MarkerSize', 3.8, 'MarkerFaceColor', 'white', ...
    'DisplayName', 'CGYRO $\gamma$');
xlabel(axisHandle, '$\eta_i$', 'Interpreter', 'latex');
ylabel(axisHandle, '$\omega R_0/v_{ti}$', 'Interpreter', 'latex');
title(axisHandle, '(a)', 'FontWeight', 'normal');
axisHandle.XTick = etaI;
grid(axisHandle, 'on');
box(axisHandle, 'on');
legendHandle = legend(axisHandle, 'Location', 'southoutside', ...
    'NumColumns', 2, 'Interpreter', 'latex');
legendHandle.Box = 'off';
legendHandle.FontSize = 6.5;
formatAxis(axisHandle);

letters = 'bcdef';
for index = 1:numel(etaI)
    axisHandle = nexttile(layout, index + 1);
    hold(axisHandle, 'on');
    plot(axisHandle, thetaMgk{index} / pi, real(phiMgk{index}), '-', ...
        'Color', mgkColor, 'LineWidth', 1.15, 'HandleVisibility', 'off');
    plot(axisHandle, thetaCgyro{index} / pi, real(phiCgyro{index}), '-', ...
        'Color', cgyroColor, 'LineWidth', 1.05, 'HandleVisibility', 'off');
    plot(axisHandle, thetaMgk{index} / pi, imag(phiMgk{index}), '--', ...
        'Color', mgkColor, 'LineWidth', 1.15, 'HandleVisibility', 'off');
    plot(axisHandle, thetaCgyro{index} / pi, imag(phiCgyro{index}), '--', ...
        'Color', cgyroColor, 'LineWidth', 1.05, 'HandleVisibility', 'off');
    yline(axisHandle, 0, ':', 'Color', [0.55, 0.55, 0.55], ...
        'LineWidth', 0.55, 'HandleVisibility', 'off');
    xlim(axisHandle, [-5, 5]);
    ylim(axisHandle, [-1.05, 1.05]);
    axisHandle.XTick = [-5, 0, 5];
    axisHandle.YTick = [-1, 0, 1];
    xlabel(axisHandle, '$\theta/\pi$', 'Interpreter', 'latex');
    if index == 1 || index == 3
        ylabel(axisHandle, '$\phi/\max|\phi|$', 'Interpreter', 'latex');
    end
    title(axisHandle, sprintf('(%c)  $\\eta_i=%.1f$', ...
        letters(index), etaI(index)), 'Interpreter', 'latex', ...
        'FontWeight', 'normal');
    grid(axisHandle, 'on');
    box(axisHandle, 'on');
    formatAxis(axisHandle);
end

exportgraphics(figureHandle, outputFile, 'ContentType', 'vector', ...
    'BackgroundColor', 'white');
close(figureHandle);
end

function formatAxis(axisHandle)
axisHandle.FontName = 'Times New Roman';
axisHandle.FontSize = 7.2;
axisHandle.LineWidth = 0.6;
axisHandle.TickDir = 'out';
axisHandle.GridAlpha = 0.12;
axisHandle.MinorGridAlpha = 0.08;
end
