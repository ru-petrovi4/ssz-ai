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

                //double minEnergy = Double.MaxValue;
                //int failCount = 0;
                for (; ; )
                {
                    await Model.ReorderMemoriesAsync(_random, cancellationToken, () =>
                    {
                        Dispatcher.UIThread.Invoke(() =>
                        {
                            Refresh_ImagesSet();
                        });
                        return Task.CompletedTask;
                    });

                    //await Model.AddNoizeAsync(400, _random, cancellationToken, () =>
                    //{
                    //    Dispatcher.UIThread.Invoke(() =>
                    //    {
                    //        Refresh_ImagesSet();
                    //    });
                    //    return Task.CompletedTask;
                    //});

                    //await Model.ReorderMemoriesAsync(100, _random, cancellationToken, () =>
                    //{
                    //    Dispatcher.UIThread.Invoke(() =>
                    //    {
                    //        Refresh_ImagesSet();
                    //    });
                    //    return Task.CompletedTask;
                    //});

                    //var energy = Model.GetEnergy();
                    //Model.LoggersSet.LoggerAndUserFriendlyLogger.LogInformation($"Energy {energy}.");
                    //if (energy < minEnergy)
                    //{
                    //    minEnergy = energy;
                    //    failCount = 0;
                    //}
                    //else
                    //{
                    //    failCount += 1;
                    //}

                    //if (failCount > 3)
                        break;
                }
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

    #endregion

    private void Reset()
    {
        var constants = Model02.Constants;
        GetDataFromControls(constants); 

        _random = new Random();

        Model = new Model02();

        Model.Cortex = new Models.CortexVisualisationModel.Cortex(Model02.Constants, Model.LoggersSet.LoggerAndUserFriendlyLogger);
        Model.Cortex.GenerateOwnedData(_random);
        Model.Cortex.Prepare();
    }

    private void Refresh_ImagesSet()
    {
        ImagesSet1_TextBlock.Text = Model.Cortex.Temp_InputCurrentDesc;
        ImagesSet1.MainItemsControl.ItemsSource = Model.GetImageWithDescs();
    }

    private Random _random = null!;

    private CancellationTokenSource? _cancellationTokenSource;

    private Task? _curentLongRunningTask;
}