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
using Ssz.AI.Models.ImageProcessingModel;
using Avalonia.Threading;
using System.Threading;
using MathNet.Numerics.Random;
using Tensorflow.Keras.Saving.SavedModel;

namespace Ssz.AI.Views.ImageProcessingModelViews;

public partial class Model01View : UserControl
{
    public Model01View()
    {
        InitializeComponent();

        if (Design.IsDesignMode)
            return;

        var constants = Model01.Constants;
        SetDataToControls(constants);

        LevelScrollBar0.ValueChanged += (s, e) => GetDataFromControls(constants);
        LevelScrollBar1.ValueChanged += (s, e) => GetDataFromControls(constants);
        LevelScrollBar2.ValueChanged += (s, e) => GetDataFromControls(constants);
        LevelScrollBar3.ValueChanged += (s, e) => GetDataFromControls(constants);
        LevelScrollBar4.ValueChanged += (s, e) => GetDataFromControls(constants);

        ColorLowScrollBar.ValueChanged += (s, e) => Refresh_ImagesSet();
        ColorHighScrollBar.ValueChanged += (s, e) => Refresh_ImagesSet();

        Reset();
        Refresh_ImagesSet();

        //Model = new Model01();
        //Task.Run(() =>
        //{
        //    Model.StemInputText();
        //});
    }

    public Model01 Model = null!;

    private void SetDataToControls(Model01.ModelConstants constants)
    {
        LevelScrollBar0.Value = constants.K0;
        //LevelScrollBar1.Value = constants.K1;
        LevelScrollBar2.Value = constants.K2;
        LevelScrollBar3.Value = constants.K3;
        LevelScrollBar4.Value = constants.K4;

        EnergyThreshold.IsChecked = constants.TotalEnergyThreshold;
        SingleMemory.IsChecked = constants.SingleMemory;

        ((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[0].Value = constants.PositiveK[1];
        ((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[1].Value = constants.PositiveK[2];
        ((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[2].Value = constants.PositiveK[3];

        ((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[0].Value = constants.NegativeK[1];
        ((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[1].Value = constants.NegativeK[2];
        ((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[2].Value = constants.NegativeK[3];
    }

    private void GetDataFromControls(Model01.ModelConstants constants)
    {
        constants.K0 = (float)LevelScrollBar0.Value;
        //constants.K1 = (float)LevelScrollBar1.Value;
        constants.K2 = (float)LevelScrollBar2.Value;
        constants.K3 = (float)LevelScrollBar3.Value;
        constants.K4 = (float)LevelScrollBar4.Value;
        
        constants.TotalEnergyThreshold = EnergyThreshold.IsChecked == true;
        constants.SingleMemory = SingleMemory.IsChecked == true;

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
        Helpers.SerializationHelper.SaveToFile(Model01.FileName_Cortex, Model.Cortex, null, Model.Logger);
    }

    private void LoadCortex_OnClick(object? sender, RoutedEventArgs args)
    {
        Helpers.SerializationHelper.LoadFromFileIfExists(Model01.FileName_Cortex, Model.Cortex, null, Model.Logger);
        Model.Cortex.Prepare(Model.LeftEye, Model.RightEye, _random);

        Refresh_ImagesSet();
    }

    private void SingleMemory_OnClick(object? sender, RoutedEventArgs args)
    {
        GetDataFromControls(Model01.Constants);
    }

    private void EnergyThreshold_OnClick(object? sender, RoutedEventArgs args)
    {
        GetDataFromControls(Model01.Constants);
    }

    private void PutInitialMemoriesPinwheel_OnClick(object? sender, RoutedEventArgs args)
    {
        Model.PutMemories_Pinwheel(_random, inMiniColumn_CortexMemoriesCount: 1);

        Refresh_ImagesSet();
    }

    private void PutInitialMemoriesRandom_OnClick(object? sender, RoutedEventArgs args)
    {
        if (Model01.Constants.SingleMemory)
            Model.PutMemories_Random_SingleMemory(_random, cortexMemoriesCount: 1);
        else
            Model.PutMemories_Random_MultiMemory(_random, cortexMemoriesCount: 1);

        Refresh_ImagesSet();
    }

    private async void AddNoize_OnClick(object? sender, RoutedEventArgs args)
    {
        if (_curentLongRunningTask is not null)
            return;
        _cancellationTokenSource = new CancellationTokenSource();
        var cancellationToken = _cancellationTokenSource.Token;

        _curentLongRunningTask = Task.Run(async () =>
        {
            try
            {
                Model.Logger.LogInformation("AddNoize Started.");

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
                Model.Logger.LogInformation("AddNoize Cancelled.");
            }

            Model.Logger.LogInformation("AddNoize Finished.");
        });
        await _curentLongRunningTask;
        _curentLongRunningTask = null;

        Refresh_ImagesSet();
    }

    private void Flood_OnClick(object? sender, RoutedEventArgs args)
    {
        Model.Flood(_random, Model.Cortex.MiniColumns);

        Refresh_ImagesSet();
    }

    private async void StartReorderMemories_OnClick(object? sender, RoutedEventArgs args)
    {
        if (_curentLongRunningTask is not null)
            return;
        _cancellationTokenSource = new CancellationTokenSource();
        var cancellationToken = _cancellationTokenSource.Token;

        _curentLongRunningTask = Task.Run(async () =>
        {
            try
            {
                Model.Logger.LogInformation("ReorderMemories Started.");

                await Model.ReorderMemories_MultiMemoryAsync(
                    _random,
                    cancellationToken, 
                    () =>
                    {
                        Dispatcher.UIThread.Invoke(() =>
                        {
                            Refresh_ImagesSet();
                        });
                        return Task.CompletedTask;
                    },
                    Model.Cortex.MiniColumns,
                    epochCount: 1000);
            }
            catch (OperationCanceledException)
            {
                Model.Logger.LogInformation("ReorderMemories Cancelled.");
            }

            Model.Logger.LogInformation("ReorderMemories Finished.");
        });
        await _curentLongRunningTask;
        _curentLongRunningTask = null;

        Refresh_ImagesSet();
    }

    private async void StartProcessSomIdeal1_OnClick(object? sender, RoutedEventArgs args)
    {
        await Model.ProcessSomIdealNAsync(0.01f, _random, CancellationToken.None, () => Task.CompletedTask);

        Refresh_ImagesSet();
    }

    private async void StartProcessSomIdealN_OnClick(object? sender, RoutedEventArgs args)
    {
        if (_curentLongRunningTask is not null)
            return;
        _cancellationTokenSource = new CancellationTokenSource();
        var cancellationToken = _cancellationTokenSource.Token;

        float epochs = new Any(await DialogHelper.GetValueFromUserAsync("epochs", defaultValue: @"1")).ValueAsSingle(false);

        _curentLongRunningTask = Task.Run(async () =>
        {
            try
            {
                Model.Logger.LogInformation("StartProcessSomIdealN Started.");

                await Model.ProcessSomIdealNAsync(epochs, _random, cancellationToken, () =>
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
                Model.Logger.LogInformation("StartProcessSomIdealN Cancelled.");
            }
            Model.Logger.LogInformation("StartProcessSomIdealN Finished.");
        });
        await _curentLongRunningTask;
        _curentLongRunningTask = null;

        Refresh_ImagesSet();
    }

    private async void StartProcessSom1_OnClick(object? sender, RoutedEventArgs args)
    {
        await Model.ProcessSomNAsync(0.01f, _random, CancellationToken.None, () => Task.CompletedTask);

        Refresh_ImagesSet();
    }

    private async void StartProcessSomN_OnClick(object? sender, RoutedEventArgs args)
    {
        if (_curentLongRunningTask is not null)
            return;
        _cancellationTokenSource = new CancellationTokenSource();
        var cancellationToken = _cancellationTokenSource.Token;

        float epochs = new Any(await DialogHelper.GetValueFromUserAsync("epochs", defaultValue: @"1")).ValueAsSingle(false);

        _curentLongRunningTask = Task.Run(async () =>
        {
            try
            {
                Model.Logger.LogInformation("StartProcessSomN Started.");

                await Model.ProcessSomNAsync(epochs, _random, cancellationToken, () =>
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
                Model.Logger.LogInformation("StartProcessSomN Cancelled.");
            }
            Model.Logger.LogInformation("StartProcessSomN Finished.");
        });
        await _curentLongRunningTask;
        _curentLongRunningTask = null;

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
            return;
        _cancellationTokenSource = new CancellationTokenSource();
        var cancellationToken = _cancellationTokenSource.Token;

        float cortexMemoriesCount = new Any(await DialogHelper.GetValueFromUserAsync("cortexMemoriesCount", defaultValue: @"1")).ValueAsSingle(false);

        _curentLongRunningTask = Task.Run(async () =>
        {
            try
            {
                Model.Logger.LogInformation("StartProcessN Started.");

                await Model.ProcessNAsync(cortexMemoriesCount, _random, cancellationToken, () =>
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
                Model.Logger.LogInformation("StartProcessN Cancelled.");
            }
            Model.Logger.LogInformation("StartProcessN Finished.");
        });
        await _curentLongRunningTask;
        _curentLongRunningTask = null;

        Refresh_ImagesSet();
    }

    //private async void StartProcessScript_OnClick(object? sender, RoutedEventArgs args)
    //{
    //    if (_curentLongRunningTask is not null)
    //        await _curentLongRunningTask;
    //    _cancellationTokenSource = new CancellationTokenSource();
    //    var cancellationToken = _cancellationTokenSource.Token;

    //    _curentLongRunningTask = Task.Run(async () =>
    //    {
    //        try
    //        {
    //            Model.LoggersSet.LoggerAndUserFriendlyLogger.LogInformation("ProcessScript Started.");

    //            var constants = Model01.Constants;

    //            BestPinwheelSettings bestPinwheelSettings = new();

    //            int interationN = 0;
    //            for (float pk1 = 0.05f; pk1 < 0.17f; pk1 += 0.01f)
    //                for (float pk2 = 0.005f; pk2 < pk1; pk2 += 0.005f)
    //                    for (float nk1 = pk1; nk1 <= pk1; nk1 += 0.01f)
    //                        for (float nk2 = 0.005f; nk2 < nk1; nk2 += 0.005f)
    //                        //for (float k3 = 0.0f; k3 <= 0.125f; k3 += 0.005f)                    
    //                        {
    //                            interationN += 1;

    //                            //float pk1 = 0.14f;
    //                            //float pk2 = 0.125f;
    //                            //float nk1 = 0.14f;
    //                            //float nk2 = 0.125f;
    //                            float k3 = 0.015f;

    //                            constants.PositiveK[1] = pk1;
    //                            constants.PositiveK[2] = pk2;
    //                            constants.PositiveK[3] = k3;
    //                            constants.NegativeK[1] = nk1;
    //                            constants.NegativeK[2] = nk2;
    //                            constants.NegativeK[3] = k3;

    //                            Model = new Model01();

    //                            Model.Cortex = new Models.ImageProcessingModel.Cortex(Model01.Constants, Model.LoggersSet.LoggerAndUserFriendlyLogger);
    //                            Model.Cortex.GenerateOwnedData(_random, onlyCeneterHypercolumn: true);
    //                            Model.Cortex.Prepare();

    //                            await Model.ProcessNAsync(900, _random, cancellationToken, () =>
    //                            {
    //                                return Task.CompletedTask;
    //                            });

    //                            await Model.ReorderMemoriesAsync(_random, cancellationToken, () =>
    //                            {
    //                                return Task.CompletedTask;
    //                            });

    //                            float pinwheellIndex = Model.GetPinwheelIndex(_random, Model.Cortex.MiniColumns);
    //                            if (pinwheellIndex > bestPinwheelSettings.MaxPinwheelIndex)
    //                            {
    //                                bestPinwheelSettings.MaxPinwheelIndex = pinwheellIndex;
    //                                bestPinwheelSettings.Pk1 = pk1;
    //                                bestPinwheelSettings.Pk2 = pk2;
    //                                bestPinwheelSettings.Pk3 = k3;
    //                                bestPinwheelSettings.Nk1 = nk1;
    //                                bestPinwheelSettings.Nk2 = nk2;
    //                                bestPinwheelSettings.Nk3 = k3;
    //                            }

    //                            Model.LoggersSet.LoggerAndUserFriendlyLogger.LogInformation(CsvHelper.FormatForCsv(
    //                                @",",
    //                                [ interationN,
    //                                bestPinwheelSettings.MaxPinwheelIndex,
    //                                bestPinwheelSettings.Pk1,
    //                                bestPinwheelSettings.Pk2,
    //                                bestPinwheelSettings.Pk3,
    //                                bestPinwheelSettings.Nk1,
    //                                bestPinwheelSettings.Nk2,
    //                                bestPinwheelSettings.Nk3,
    //                                "Current",
    //                                pinwheellIndex,
    //                                pk1,
    //                                pk2,
    //                                k3,
    //                                nk1,
    //                                nk2,
    //                                k3 ]));
    //                        }
    //        }
    //        catch (OperationCanceledException)
    //        {
    //            Model.LoggersSet.LoggerAndUserFriendlyLogger.LogInformation("ProcessScript Cancelled.");
    //        }
    //        Model.LoggersSet.LoggerAndUserFriendlyLogger.LogInformation("ProcessScript Finished.");
    //    });
    //    await _curentLongRunningTask;
    //_curentLongRunningTask = null;

    //    Refresh_ImagesSet();
    //}

    private async void StartProcessScript_OnClick(object? sender, RoutedEventArgs args)
    {
        if (_curentLongRunningTask is not null)
            return;
        _cancellationTokenSource = new CancellationTokenSource();
        var cancellationToken = _cancellationTokenSource.Token;

        _curentLongRunningTask = Task.Run(async () =>
        {
            try
            {
                Model.Logger.LogInformation("ProcessScript Started.");

                var constants = Model01.Constants;

                int count = 100;
                for (int it = 0; it < count; it += 1)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    Model.Flood(_random, Model.Cortex.MiniColumns);

                    await Model.ProcessNAsync(3, _random, cancellationToken, () =>
                    {
                        Dispatcher.UIThread.Invoke(() =>
                        {
                            Refresh_ImagesSet();
                        });
                        return Task.CompletedTask;
                    });

                    await Model.ReorderMemories_MultiMemoryAsync(
                        _random, 
                        cancellationToken,
                        () =>
                        {
                            Dispatcher.UIThread.Invoke(() =>
                            {
                                Refresh_ImagesSet();
                            });
                            return Task.CompletedTask;
                        },
                        Model.Cortex.MiniColumns,
                        epochCount: 30);
                }
            }
            catch (OperationCanceledException)
            {
                Model.Logger.LogInformation("ProcessScript Cancelled.");
            }
            Model.Logger.LogInformation("ProcessScript Finished.");
        });
        await _curentLongRunningTask;
        _curentLongRunningTask = null;

        Refresh_ImagesSet();
    }

    private async void StartProcessScript_OptimizeSettings_OnClick(object? sender, RoutedEventArgs args)
    {
        if (_curentLongRunningTask is not null)
            return;
        _cancellationTokenSource = new CancellationTokenSource();
        var cancellationToken = _cancellationTokenSource.Token;

        _curentLongRunningTask = Task.Run(async () =>
        {
            try
            {
                Model.Logger.LogInformation("ProcessScript Started.");

                var constants = Model01.Constants;

                BestSettings bestSettings = new();

                int interationN = 0;
                for (float k3 = 0.00f; k3 < 0.33; k3 += 0.05f)
                {
                    interationN += 1;

                    constants.K3 = k3;

                    float index = 0.0f;
                    int count = 100;
                    for (int it = 0; it < count; it += 1)
                    {
                        Model = new Model01(_random, onlyCenterHypercolumn: true);                        

                        Model.PutMemories_Random_MultiMemory(_random, cortexMemoriesCount: 6);

                        await Model.ReorderMemories_MultiMemoryAsync(
                            _random, 
                            cancellationToken,                            
                            () =>
                            {
                                return Task.CompletedTask;
                            },
                            Model.Cortex.MiniColumns,
                            epochCount: 30);

                        index += ((Model.GetPinwheelIndex(_random, Model.Cortex.MiniColumns, hypercolumnIndex: 0) > 4.5) ? 1.0f : 0.0f);
                    }
                    index = index / count;
                    if (index > bestSettings.MaxIndex)
                    {
                        bestSettings.MaxIndex = index;
                        //bestSettings.Pk1 = pk1;
                        //bestSettings.Pk2 = pk2;
                        bestSettings.Pk3 = k3;
                        //bestSettings.Nk1 = nk1;
                        //bestSettings.Nk2 = nk2;
                        //bestSettings.Nk3 = k3;
                    }

                    Model.Logger.LogInformation(CsvHelper.FormatForCsv(
                                    @",",
                                    [ interationN,
                                    bestSettings.MaxIndex,
                                    bestSettings.Pk3,
                                    0.0f,
                                    0.0f,
                                    0.0f,
                                    0.0f,
                                    0.0f,
                                    "Current",
                                    index,
                                    k3,
                                    0.0f,
                                    0.0f,
                                    0.0f,
                                    0.0f,
                                    0.0f ]));
                }
            }
            catch (OperationCanceledException)
            {
                Model.Logger.LogInformation("ProcessScript Cancelled.");
            }
            Model.Logger.LogInformation("ProcessScript Finished.");
        });
        await _curentLongRunningTask;
        _curentLongRunningTask = null;

        Refresh_ImagesSet();
    }

    private void VisualizesScriptResults_OnClick(object? sender, RoutedEventArgs args)
    {
        ImagesSet1.MainItemsControl.ItemsSource = Visualisation.VisualizeKSearch();
    }

    #endregion

    private void Reset()
    {
        var constants = Model01.Constants;
        GetDataFromControls(constants);

        _random = new Random();

        Model = new Model01(_random, OnlyCenterHyperColumn);
    }

    private void Refresh_ImagesSet()
    {
        ImagesSet1_TextBlock.Text = Model.Cortex.Temp_InputCurrentDesc;
        ImagesSet1.MainItemsControl.ItemsSource = Model.GetImageWithDescs(
            _random, 
            ColorLowScrollBar.Value,
            ColorHighScrollBar.Value);
    }

    private const bool OnlyCenterHyperColumn = true;

    private Random _random = null!;

    private CancellationTokenSource? _cancellationTokenSource;

    private Task? _curentLongRunningTask;

    private class BestSettings
    {
        public float MaxIndex = float.MinValue;

        public float Pk1;
        public float Pk2;
        public float Pk3;
        public float Nk1;
        public float Nk2;
        public float Nk3;
    }
}