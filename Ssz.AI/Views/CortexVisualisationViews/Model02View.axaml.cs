using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Interactivity;
using Microsoft.AspNetCore.Identity;
using Microsoft.Extensions.Logging;
using MsBox.Avalonia;
using Ssz.AI.Helpers;
using Ssz.AI.Models;
using Ssz.Utils;
using System;
using System.Collections.Generic;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using System.Threading.Tasks;
using Ssz.AI.Models.CortexVisualisationModel;
using Avalonia.Threading;
using System.Threading;
using MathNet.Numerics.Random;

namespace Ssz.AI.Views.CortexVisualisationViews;

public partial class Model02View : UserControl
{
    public Model02View()
    {
        InitializeComponent();

        if (Design.IsDesignMode)
            return;

        var constants = Model02.Constants;
        SetDataToControls(constants);

        LevelScrollBar0.ValueChanged += (s, e) => GetDataFromControls(constants);
        LevelScrollBar1.ValueChanged += (s, e) => GetDataFromControls(constants);
        LevelScrollBar2.ValueChanged += (s, e) => GetDataFromControls(constants);
        LevelScrollBar3.ValueChanged += (s, e) => GetDataFromControls(constants);
        LevelScrollBar4.ValueChanged += (s, e) => GetDataFromControls(constants);

        Reset();
        Refresh_ImagesSet();

        //Model = new Model02();
        //Task.Run(() =>
        //{
        //    Model.StemInputText();
        //});
    }

    public Model02 Model = null!;

    private void SetDataToControls(Model02.ModelConstants constants)
    {
        LevelScrollBar0.Value = constants.K0;
        //LevelScrollBar1.Value = constants.K1;
        LevelScrollBar2.Value = constants.K2;
        //LevelScrollBar3.Value = constants.K3;
        LevelScrollBar4.Value = constants.K4;

        SuperactivityThreshold.IsChecked = constants.SuperactivityThreshold;

        ((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[0].Value = constants.PositiveK[1];
        ((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[1].Value = constants.PositiveK[2];
        ((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[2].Value = constants.PositiveK[3];

        ((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[0].Value = constants.NegativeK[1];
        ((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[1].Value = constants.NegativeK[2];
        ((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[2].Value = constants.NegativeK[3];
    }

    private void GetDataFromControls(Model02.ModelConstants constants)
    {
        constants.K0 = (float)LevelScrollBar0.Value;
        //constants.K1 = (float)LevelScrollBar1.Value;
        constants.K2 = (float)LevelScrollBar2.Value;
        //constants.K3 = (float)LevelScrollBar3.Value;
        constants.K4 = (float)LevelScrollBar4.Value;

        constants.SuperactivityThreshold = SuperactivityThreshold.IsChecked == true;

        constants.PositiveK[1] = (float)((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[0].Value;
        constants.PositiveK[2] = (float)((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[1].Value;
        constants.PositiveK[3] = (float)((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[2].Value;

        constants.NegativeK[1] = (float)((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[0].Value;
        constants.NegativeK[2] = (float)((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[1].Value;
        constants.NegativeK[3] = (float)((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[2].Value;
    }

    #region Buttons Handlers

    private void Reset_OnClick(object? sender, RoutedEventArgs args)
    {
        Reset();

        Refresh_ImagesSet();
    }

    private void StopLongOperation_OnClick(object? sender, RoutedEventArgs args)
    {
        _cancellationTokenSource?.Cancel();
    }

    private void SaveCortex_OnClick(object? sender, RoutedEventArgs args)
    {
        Helpers.SerializationHelper.SaveToFile(Model02.FileName_Cortex, Model.Cortex, null, Model.LoggersSet.LoggerAndUserFriendlyLogger);
    }

    private void LoadCortex_OnClick(object? sender, RoutedEventArgs args)
    {
        Helpers.SerializationHelper.LoadFromFileIfExists(Model02.FileName_Cortex, Model.Cortex, null, Model.LoggersSet.LoggerAndUserFriendlyLogger);
        Model.Cortex.Prepare();
    }

    private void SuperactivityThreshold_OnClick(object? sender, RoutedEventArgs args)
    {
        GetDataFromControls(Model02.Constants);
    }

    private void PutInitialMemoriesPinwheel_OnClick(object? sender, RoutedEventArgs args)
    {
        Model.PutInitialMemoriesPinwheel(_random, isRandom: false);

        Refresh_ImagesSet();
    }

    private void PutInitialMemoriesRandom_OnClick(object? sender, RoutedEventArgs args)
    {
        Model.PutInitialMemoriesPinwheel(_random, isRandom: true);

        Refresh_ImagesSet();
    }

    private async void AddNoize_OnClick(object? sender, RoutedEventArgs args)
    {
        if (_curentLongRunningTask is not null)
            await _curentLongRunningTask;
        _cancellationTokenSource = new CancellationTokenSource();
        var cancellationToken = _cancellationTokenSource.Token;

        _curentLongRunningTask = Task.Run(async () =>
        {
            try
            {
                Model.LoggersSet.LoggerAndUserFriendlyLogger.LogInformation("AddNoize Started.");

                await Model.AddNoizeAsync(20, _random, cancellationToken, () =>
                {
                    Dispatcher.UIThread.Invoke(() =>
                    {
                        Refresh_ImagesSet();
                    });
                    return Task.CompletedTask;
                });
            }
            catch (OperationCanceledException)
            {
                Model.LoggersSet.LoggerAndUserFriendlyLogger.LogInformation("AddNoize Cancelled.");
            }

            Model.LoggersSet.LoggerAndUserFriendlyLogger.LogInformation("AddNoize Finished.");
        });
        await _curentLongRunningTask;

        Refresh_ImagesSet();
    }

    private async void Flood_OnClick(object? sender, RoutedEventArgs args)
    {
        Model.Flood(_random, Model.Cortex.MiniColumns);

        Refresh_ImagesSet();
    }

    private async void StartReorderMemories_OnClick(object? sender, RoutedEventArgs args)
    {
        if (_curentLongRunningTask is not null)
            await _curentLongRunningTask;
        _cancellationTokenSource = new CancellationTokenSource();
        var cancellationToken = _cancellationTokenSource.Token;

        _curentLongRunningTask = Task.Run(async () =>
        {
            try
            {
                Model.LoggersSet.LoggerAndUserFriendlyLogger.LogInformation("ReorderMemories Started.");

                await Model.ReorderMemoriesAsync(_random, cancellationToken, () =>
                {
                    Dispatcher.UIThread.Invoke(() =>
                    {
                        Refresh_ImagesSet();
                    });
                    return Task.CompletedTask;
                });
            }
            catch (OperationCanceledException)
            {
                Model.LoggersSet.LoggerAndUserFriendlyLogger.LogInformation("ReorderMemories Cancelled.");
            }

            Model.LoggersSet.LoggerAndUserFriendlyLogger.LogInformation("ReorderMemories Finished.");
        });
        await _curentLongRunningTask;

        Refresh_ImagesSet();
    }

    private async void StartProcess1_OnClick(object? sender, RoutedEventArgs args)
    {
        await Model.ProcessNAsync(1, _random, CancellationToken.None, () => Task.CompletedTask);

        Refresh_ImagesSet();
    }

    private async void StartProcessN_OnClick(object? sender, RoutedEventArgs args)
    {
        if (_curentLongRunningTask is not null)
            await _curentLongRunningTask;
        _cancellationTokenSource = new CancellationTokenSource();
        var cancellationToken = _cancellationTokenSource.Token;

        _curentLongRunningTask = Task.Run(async () =>
        {
            try
            {
                Model.LoggersSet.LoggerAndUserFriendlyLogger.LogInformation("StartProcessN Started.");

                await Model.ProcessNAsync(300, _random, cancellationToken, () =>
                {
                    Dispatcher.UIThread.Invoke(() =>
                    {
                        Refresh_ImagesSet();
                    });
                    return Task.CompletedTask;
                });
            }
            catch (OperationCanceledException)
            {
                Model.LoggersSet.LoggerAndUserFriendlyLogger.LogInformation("StartProcessN Cancelled.");
            }
            Model.LoggersSet.LoggerAndUserFriendlyLogger.LogInformation("StartProcessN Finished.");
        });
        await _curentLongRunningTask;

        Refresh_ImagesSet();
    }

    private async void StartProcessScript_OnClick(object? sender, RoutedEventArgs args)
    {
        if (_curentLongRunningTask is not null)
            await _curentLongRunningTask;
        _cancellationTokenSource = new CancellationTokenSource();
        var cancellationToken = _cancellationTokenSource.Token;

        _curentLongRunningTask = Task.Run(async () =>
        {
            try
            {
                Model.LoggersSet.LoggerAndUserFriendlyLogger.LogInformation("ProcessScript Started.");

                var constants = Model02.Constants;

                BestPinwheelSettings bestPinwheelSettings = new();

                int interationN = 0;
                for (float pk1 = 0.05f; pk1 < 0.17f; pk1 += 0.01f)
                    for (float pk2 = 0.005f; pk2 < pk1; pk2 += 0.005f)
                        for (float nk1 = pk1; nk1 <= pk1; nk1 += 0.01f)
                            for (float nk2 = 0.005f; nk2 < nk1; nk2 += 0.005f)
                            //for (float k3 = 0.0f; k3 <= 0.125f; k3 += 0.005f)                    
                            {
                                interationN += 1;

                                //float pk1 = 0.14f;
                                //float pk2 = 0.125f;
                                //float nk1 = 0.14f;
                                //float nk2 = 0.125f;
                                float k3 = 0.015f;

                                constants.PositiveK[1] = pk1;
                                constants.PositiveK[2] = pk2;
                                constants.PositiveK[3] = k3;
                                constants.NegativeK[1] = nk1;
                                constants.NegativeK[2] = nk2;
                                constants.NegativeK[3] = k3;

                                Model = new Model02();

                                Model.Cortex = new Models.CortexVisualisationModel.Cortex(Model02.Constants, Model.LoggersSet.LoggerAndUserFriendlyLogger);
                                Model.Cortex.GenerateOwnedData(_random, onlyCeneterHypercolumn: true);
                                Model.Cortex.Prepare();

                                await Model.ProcessNAsync(900, _random, cancellationToken, () =>
                                {
                                    return Task.CompletedTask;
                                });

                                await Model.ReorderMemoriesAsync(_random, cancellationToken, () =>
                                {
                                    return Task.CompletedTask;
                                });

                                float pinwheellIndex = Model.GetPinwheelIndex(_random, Model.Cortex.MiniColumns);
                                if (pinwheellIndex > bestPinwheelSettings.MaxPinwheelIndex)
                                {
                                    bestPinwheelSettings.MaxPinwheelIndex = pinwheellIndex;
                                    bestPinwheelSettings.Pk1 = pk1;
                                    bestPinwheelSettings.Pk2 = pk2;
                                    bestPinwheelSettings.Pk3 = k3;
                                    bestPinwheelSettings.Nk1 = nk1;
                                    bestPinwheelSettings.Nk2 = nk2;
                                    bestPinwheelSettings.Nk3 = k3;
                                }

                                Model.LoggersSet.LoggerAndUserFriendlyLogger.LogInformation(CsvHelper.FormatForCsv(
                                    @",",
                                    [ interationN,
                                    bestPinwheelSettings.MaxPinwheelIndex,
                                    bestPinwheelSettings.Pk1,
                                    bestPinwheelSettings.Pk2,
                                    bestPinwheelSettings.Pk3,
                                    bestPinwheelSettings.Nk1,
                                    bestPinwheelSettings.Nk2,
                                    bestPinwheelSettings.Nk3,
                                    "Current",
                                    pinwheellIndex,
                                    pk1,
                                    pk2,
                                    k3,
                                    nk1,
                                    nk2,
                                    k3 ]));
                            }
            }
            catch (OperationCanceledException)
            {
                Model.LoggersSet.LoggerAndUserFriendlyLogger.LogInformation("ProcessScript Cancelled.");
            }
            Model.LoggersSet.LoggerAndUserFriendlyLogger.LogInformation("ProcessScript Finished.");
        });
        await _curentLongRunningTask;

        Refresh_ImagesSet();
    }

    private void VisualizesScriptResults_OnClick(object? sender, RoutedEventArgs args)
    {
        ImagesSet1.MainItemsControl.ItemsSource = Visualisation.VisualizeKSearch();
    }

    #endregion

    private void Reset()
    {
        var constants = Model02.Constants;
        GetDataFromControls(constants);

        _random = new Random();

        Model = new Model02();

        Model.Cortex = new Models.CortexVisualisationModel.Cortex(Model02.Constants, Model.LoggersSet.LoggerAndUserFriendlyLogger);
        Model.Cortex.GenerateOwnedData(_random, onlyCeneterHypercolumn: true);
        Model.Cortex.Prepare();
    }

    private void Refresh_ImagesSet()
    {
        ImagesSet1_TextBlock.Text = Model.Cortex.Temp_InputCurrentDesc;
        ImagesSet1.MainItemsControl.ItemsSource = Model.GetImageWithDescs(_random);
    }

    private Random _random = null!;

    private CancellationTokenSource? _cancellationTokenSource;

    private Task? _curentLongRunningTask;

    private class BestPinwheelSettings
    {
        public float MaxPinwheelIndex = float.MinValue;

        public float Pk1;
        public float Pk2;
        public float Pk3;
        public float Nk1;
        public float Nk2;
        public float Nk3;
    }
}