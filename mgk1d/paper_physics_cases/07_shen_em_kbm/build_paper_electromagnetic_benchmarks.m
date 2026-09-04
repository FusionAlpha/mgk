function outputs = build_paper_electromagnetic_benchmarks(target)
% Replot the two principal electromagnetic benchmarks in manuscript style.

if nargin < 1 || isempty(target)
    target = "all";
else
    target = lower(string(target));
end
assert(ismember(target, ["all", "xie", "shen"]), ...
    'Expected target to be all, xie, or shen.');

analysisDirectory = fileparts(mfilename('fullpath'));
analysisRoot = fileparts(analysisDirectory);
projectRoot = fileparts(analysisRoot);
figureDirectory = fullfile(projectRoot, 'figures');
if ~isfolder(figureDirectory)
    mkdir(figureDirectory);
end

xiePdf = fullfile(figureDirectory, ...
    'xie2016_electromagnetic_cbc_benchmark.pdf');
shenPdf = fullfile(figureDirectory, ...
    'shen2025_electromagnetic_kbm_benchmark.pdf');

outputs = struct('xie2016', xiePdf, 'shen2025', shenPdf);
if target == "all" || target == "xie"
    plotXieBenchmark(analysisRoot, xiePdf);
    fprintf('wrote %s\n', xiePdf);
end
if target == "all" || target == "shen"
    plotShenBenchmark(analysisRoot, shenPdf);
    fprintf('wrote %s\n', shenPdf);
end
end

function plotXieBenchmark(analysisRoot, outputFile)
source = load(fullfile(analysisRoot, ...
    'xie2016_em_cbc_gpu_open_10pi_nt161_dense002pct_scan.mat'), ...
    'data', 'reference');
mgk = source.data;
cgyro = source.reference;
gs2 = readtable(fullfile(analysisRoot, ...
    'xie2016_fig1_gs2_digitized.csv'));

required = {'betaPercent', 'dominantBranch', 'omegaR0OverCs', ...
    'gammaR0OverCs'};
assert(all(ismember(required, mgk.Properties.VariableNames)), ...
    'The Xie MGK scan is missing required columns.');
assert(all(diff(mgk.betaPercent) > 0), ...
    'The Xie MGK beta grid must be strictly increasing.');
assert(all(ismember(["ITG", "TEM", "KBM"], ...
    unique(string(mgk.dominantBranch)))), ...
    'The Xie scan does not contain all three dominant branches.');

mgkColor = [0.00, 0.45, 0.70];
cgyroColor = [0.90, 0.62, 0.00];
gs2Color = [0.00, 0.62, 0.45];

figureHandle = figure('Visible', 'off', 'Color', 'white', ...
    'Units', 'inches', 'Position', [0.5, 0.5, 7.15, 2.85], ...
    'Renderer', 'painters');
layout = tiledlayout(figureHandle, 1, 2, ...
    'TileSpacing', 'compact', 'Padding', 'compact');

axisOmega = nexttile(layout, 1);
hold(axisOmega, 'on');
hMgk = plotDominantSegments(axisOmega, mgk.betaPercent, ...
    mgk.omegaR0OverCs, string(mgk.dominantBranch), mgkColor, 'MGK Model-a');
hCgyro = plot(axisOmega, 100 * cgyro.beta, cgyro.cgyroOmega, 'd', ...
    'LineStyle', 'none', 'Color', cgyroColor, ...
    'MarkerFaceColor', 'white', 'MarkerSize', 3.8, 'LineWidth', 0.9, ...
    'DisplayName', 'CGYRO Model-a');
validOmega = isfinite(gs2.omegaR0OverCs);
hGs2 = plot(axisOmega, 100 * gs2.betaElectron(validOmega), ...
    gs2.omegaR0OverCs(validOmega), 'o', 'LineStyle', 'none', ...
    'Color', gs2Color, 'MarkerFaceColor', 'white', ...
    'MarkerSize', 4.1, 'LineWidth', 0.9, ...
    'DisplayName', 'GS2 Model-b');
yline(axisOmega, 0, ':', 'Color', [0.55, 0.55, 0.55], ...
    'LineWidth', 0.6, 'HandleVisibility', 'off');
xlabel(axisOmega, '$\beta_e$ (\%)', 'Interpreter', 'latex');
ylabel(axisOmega, '$\omega_r R_0/c_s$', 'Interpreter', 'latex');
title(axisOmega, '(a)', 'FontWeight', 'normal');
xlim(axisOmega, [0, 2]);
axisOmega.XTick = 0:0.5:2;
ylim(axisOmega, [-2.75, 0.60]);
formatAxis(axisOmega);

axisGamma = nexttile(layout, 2);
hold(axisGamma, 'on');
plotDominantSegments(axisGamma, mgk.betaPercent, mgk.gammaR0OverCs, ...
    string(mgk.dominantBranch), mgkColor, 'MGK Model-a');
plot(axisGamma, 100 * cgyro.beta, cgyro.cgyroGamma, 'd', ...
    'LineStyle', 'none', 'Color', cgyroColor, ...
    'MarkerFaceColor', 'white', 'MarkerSize', 3.8, 'LineWidth', 0.9, ...
    'HandleVisibility', 'off');
validGamma = isfinite(gs2.gammaR0OverCs);
plot(axisGamma, 100 * gs2.betaElectron(validGamma), ...
    gs2.gammaR0OverCs(validGamma), 'o', 'LineStyle', 'none', ...
    'Color', gs2Color, 'MarkerFaceColor', 'white', ...
    'MarkerSize', 4.1, 'LineWidth', 0.9, 'HandleVisibility', 'off');
xlabel(axisGamma, '$\beta_e$ (\%)', 'Interpreter', 'latex');
ylabel(axisGamma, '$\gamma R_0/c_s$', 'Interpreter', 'latex');
title(axisGamma, '(b)', 'FontWeight', 'normal');
xlim(axisGamma, [0, 2]);
axisGamma.XTick = 0:0.5:2;
ylim(axisGamma, [0, 1.38]);
formatAxis(axisGamma);

legendHandle = legend(axisOmega, [hMgk, hCgyro, hGs2], ...
    'Location', 'southoutside', 'Orientation', 'horizontal', ...
    'NumColumns', 3);
legendHandle.Layout.Tile = 'south';
legendHandle.Box = 'off';
legendHandle.FontSize = 7.0;

exportFigure(figureHandle, outputFile);
end

function handle = plotDominantSegments(axisHandle, x, y, branch, color, label)
change = [true; branch(2:end) ~= branch(1:(end - 1))];
segment = cumsum(change);
handle = gobjects(1);
for index = 1:max(segment)
    mask = segment == index;
    visibility = 'off';
    displayName = '';
    if index == 1
        visibility = 'on';
        displayName = label;
    end
    current = plot(axisHandle, x(mask), y(mask), '-', ...
        'Color', color, 'LineWidth', 1.45, ...
        'HandleVisibility', visibility, 'DisplayName', displayName);
    if index == 1
        handle = current;
    end
end
end

function plotShenBenchmark(analysisRoot, outputFile)
benchmarkDirectory = fullfile(analysisRoot, 'shen2025_em_benchmark');
mgk = readtable(fullfile(benchmarkDirectory, ...
    'mgk_matched_beta_scan_08_25_step01pct.csv'));
savedScan = load(fullfile(benchmarkDirectory, ...
    'mgk_matched_beta_scan_08_25_step01pct.mat'), 'baseConfig');
config = savedScan.baseConfig;
cgyro = readtable(fullfile(analysisRoot, ...
    'cgyro_shen2025_matched_beta_scan', 'cgyro_matched_beta_scan.csv'));
cgyro = cgyro(string(cgyro.exitCode) == "0" & ...
    cgyro.linearConverged == 1, :);

requiredMgk = {'betaPercent', 'twoFieldOmega', 'twoFieldGamma', ...
    'threeFieldOmega', 'threeFieldGamma'};
requiredCgyro = {'betaElectron', 'nField', 'omegaRoverVti', ...
    'gammaRoverVti'};
assert(all(ismember(requiredMgk, mgk.Properties.VariableNames)), ...
    'The Shen MGK scan is missing required columns.');
assert(all(ismember(requiredCgyro, cgyro.Properties.VariableNames)), ...
    'The Shen CGYRO scan is missing required columns.');
assert(all(ismember([2, 3], unique(cgyro.nField).')), ...
    'The Shen CGYRO data do not contain both field models.');
aspectRatio = config.physical.minorRadius / config.physical.majorRadius;
assert(abs(aspectRatio - 0.0018) < 1e-12, ...
    'The Shen benchmark must use the large-aspect-ratio value r/R0=0.0018.');
assert(strcmpi(config.model.parallelBoundary, 'periodic'), ...
    'The matched Shen benchmark must use the periodic parallel boundary.');
assert(abs(config.grid.thetaMin + 5 * pi) < 1e-12 && ...
    abs(config.grid.thetaMax - 5 * pi) < 1e-12 && ...
    config.grid.numTheta == 241, ...
    'The matched Shen benchmark must use [-5pi,5pi] with Ntheta=241.');

mgk = mgk(mgk.betaPercent >= 0.8, :);
twoColor = [0.00, 0.45, 0.70];
threeColor = [0.78, 0.20, 0.16];

figureHandle = figure('Visible', 'off', 'Color', 'white', ...
    'Units', 'inches', 'Position', [0.5, 0.5, 7.15, 2.85], ...
    'Renderer', 'painters');
layout = tiledlayout(figureHandle, 1, 2, ...
    'TileSpacing', 'compact', 'Padding', 'compact');

axisOmega = nexttile(layout, 1);
hold(axisOmega, 'on');
hMgkTwo = plot(axisOmega, mgk.betaPercent, mgk.twoFieldOmega, '-', ...
    'Color', twoColor, 'LineWidth', 1.45, ...
    'DisplayName', 'MGK, two fields');
hMgkThree = plot(axisOmega, mgk.betaPercent, mgk.threeFieldOmega, '--', ...
    'Color', threeColor, 'LineWidth', 1.45, ...
    'DisplayName', 'MGK, three fields');
two = cgyro.nField == 2;
three = cgyro.nField == 3;
hCgyroTwo = plot(axisOmega, 100 * cgyro.betaElectron(two), ...
    cgyro.omegaRoverVti(two), 'd', 'LineStyle', 'none', ...
    'Color', twoColor, 'MarkerFaceColor', 'white', ...
    'MarkerSize', 3.8, 'LineWidth', 0.9, ...
    'DisplayName', 'CGYRO, two fields');
hCgyroThree = plot(axisOmega, 100 * cgyro.betaElectron(three), ...
    cgyro.omegaRoverVti(three), 's', 'LineStyle', 'none', ...
    'Color', threeColor, 'MarkerFaceColor', 'white', ...
    'MarkerSize', 3.8, 'LineWidth', 0.9, ...
    'DisplayName', 'CGYRO, three fields');
xlabel(axisOmega, '$\beta_e$ (\%)', 'Interpreter', 'latex');
ylabel(axisOmega, '$\omega_r R_0/v_{\mathrm{th},i}$', ...
    'Interpreter', 'latex');
title(axisOmega, '(a)', 'FontWeight', 'normal');
xlim(axisOmega, [0.8, 2.5]);
axisOmega.XTick = [0.8, 1.0, 1.5, 2.0, 2.5];
ylim(axisOmega, [-3.55, -1.75]);
formatAxis(axisOmega);

axisGamma = nexttile(layout, 2);
hold(axisGamma, 'on');
plot(axisGamma, mgk.betaPercent, mgk.twoFieldGamma, '-', ...
    'Color', twoColor, 'LineWidth', 1.45, 'HandleVisibility', 'off');
plot(axisGamma, mgk.betaPercent, mgk.threeFieldGamma, '--', ...
    'Color', threeColor, 'LineWidth', 1.45, 'HandleVisibility', 'off');
plot(axisGamma, 100 * cgyro.betaElectron(two), cgyro.gammaRoverVti(two), ...
    'd', 'LineStyle', 'none', 'Color', twoColor, ...
    'MarkerFaceColor', 'white', 'MarkerSize', 3.8, 'LineWidth', 0.9, ...
    'HandleVisibility', 'off');
plot(axisGamma, 100 * cgyro.betaElectron(three), ...
    cgyro.gammaRoverVti(three), 's', 'LineStyle', 'none', ...
    'Color', threeColor, 'MarkerFaceColor', 'white', ...
    'MarkerSize', 3.8, 'LineWidth', 0.9, 'HandleVisibility', 'off');
xlabel(axisGamma, '$\beta_e$ (\%)', 'Interpreter', 'latex');
ylabel(axisGamma, '$\gamma R_0/v_{\mathrm{th},i}$', ...
    'Interpreter', 'latex');
title(axisGamma, '(b)', 'FontWeight', 'normal');
xlim(axisGamma, [0.8, 2.5]);
axisGamma.XTick = [0.8, 1.0, 1.5, 2.0, 2.5];
ylim(axisGamma, [0, 2.35]);
formatAxis(axisGamma);

legendHandle = legend(axisOmega, ...
    [hMgkTwo, hCgyroTwo, hMgkThree, hCgyroThree], ...
    'Location', 'southoutside', 'Orientation', 'horizontal', ...
    'NumColumns', 4);
legendHandle.Layout.Tile = 'south';
legendHandle.Box = 'off';
legendHandle.FontSize = 6.7;

exportFigure(figureHandle, outputFile);
end

function formatAxis(axisHandle)
grid(axisHandle, 'on');
box(axisHandle, 'on');
axisHandle.FontName = 'Times New Roman';
axisHandle.FontSize = 7.5;
axisHandle.LineWidth = 0.6;
axisHandle.TickDir = 'out';
axisHandle.GridAlpha = 0.12;
axisHandle.MinorGridAlpha = 0.08;
end

function exportFigure(figureHandle, outputFile)
exportgraphics(figureHandle, outputFile, 'ContentType', 'vector', ...
    'BackgroundColor', 'white');
[directory, name] = fileparts(outputFile);
exportgraphics(figureHandle, fullfile(directory, [name '.png']), ...
    'Resolution', 300, 'BackgroundColor', 'white');
close(figureHandle);
end
