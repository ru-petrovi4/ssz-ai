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
using Ssz.AI.Models.MiniColumnDetailedModel;
using Avalonia.Threading;
using System.Threading;
using MathNet.Numerics.Random;
using Tensorflow.Keras.Saving.SavedModel;
using Ssz.AI.ViewModels;
using Ssz.AI.Models.ImageProcessingModel;
using OfficeOpenXml.Table.PivotTable;
using Avalonia.Input;
using System.Text.Json;

namespace Ssz.AI.Views.MiniColumnDetailedViews;

public partial class Model01View : UserControl
{
    public Model01View()
    {
        InitializeComponent();

        if (Design.IsDesignMode)
            return;

        var constants = Model01.Constants;
        SetDataToControls(constants);

        LevelScrollBar0.ValueChanged += (s, e) =>
        {
            GetDataFromControls(constants);            
            _ = Refresh();
        };
        LevelScrollBar1.ValueChanged += (s, e) =>
        {
            GetDataFromControls(constants);
            _ = Refresh();
        };
        LevelScrollBar2.ValueChanged += (s, e) =>
        {
            GetDataFromControls(constants);
            _ = Refresh();
        };
        LevelScrollBar3.ValueChanged += (s, e) =>
        {
            GetDataFromControls(constants);
            _ = Refresh();
        };
        LevelScrollBar4.ValueChanged += (s, e) =>
        {
            GetDataFromControls(constants);
            _ = Refresh();
        };
        LevelScrollBar5.ValueChanged += (s, e) =>
        {
            GetDataFromControls(constants);
            _ = Refresh();
        };

        Reset();
        _ = Refresh();
        //if (true)
        //{
        //    Reset();
        //    Refresh_ImagesSet();
        //}
        //else
        //{
        //    var t = Script2Async();
        //}

        Unloaded += Model01View_Unloaded;

        KeyDown += (sender, e) =>
        {
            if (e.Key == Key.Right)
            {
                var sb = LevelScrollBar0!;
                double step = 0.5;
                sb.Value = Math.Min(sb.Maximum, sb.Value + step);
                e.Handled = true; // предотвращаем двойное срабатывание
            }
            else if (e.Key == Key.Left)
            {
                var sb = LevelScrollBar0!;
                double step = -0.5;
                sb.Value = Math.Min(sb.Maximum, sb.Value + step);
                e.Handled = true; // предотвращаем двойное срабатывание
            }
        };
    }

    private void Model01View_Unloaded(object? sender, RoutedEventArgs e)
    {
        Model = null;
    }

    //public async Task Script2Async()
    //{
    //    DateTime startDateTime = DateTime.Now;
    //    _random = new Random(40);
    //    var constants = Model01.Constants;
    //    GetDataFromControls(constants);

    //    for (float it = 0.4f; it < 4.0; it += 0.2f)
    //    {
    //        await Task.Run(async () =>
    //        {
    //            try
    //            {
    //                Model = null!;
    //                GC.Collect();

    //                constants.DetectorFieldOfViewRadiusPixels = it;

    //                Model = new Model01(_random, OnlyCenterHyperColumn);

    //                Model!.Logger.LogInformation("StartProcessSomIdealN Started.");

    //                await Model!.ProcessSomNAsync(epochsCount: null, _random, CancellationToken.None, () =>
    //                {
    //                    Dispatcher.UIThread.Invoke(() =>
    //                    {
    //                        Refresh_ImagesSet();
    //                    });
    //                    return Task.CompletedTask;
    //                },
    //                isIdeal: false);
    //            }
    //            catch (OperationCanceledException)
    //            {
    //                Model!.Logger.LogInformation("StartProcessSomIdealN Cancelled.");
    //            }
    //            Model!.Logger.LogInformation("StartProcessSomIdealN Finished.");
    //        });

    //        var imageWithDesc = (ImageWithDesc)(Model!.GetImageWithDescs(
    //            _random,
    //            ColorLowScrollBar.Value,
    //            ColorHighScrollBar.Value)[5]);

    //        imageWithDesc.Image!.Save(Path.Combine("Data", "Script2", FileSystemHelper.ReplaceInvalidChars($"{new Any(startDateTime).ValueAsString(false)}_{new Any(it).ValueAsString(false, "F02")}.png")));
    //    }
    //}

    private Model01? _model = null!;

    public Model01? Model
    {
        get
        {
            return _model;
        }
        set
        {
            _model?.Dispose();
            _model = value;
        }
    }

    private void SetDataToControls(Model01.ModelConstants constants)
    {
        LevelScrollBar0.Value = constants.TestGradientAngleDegrees;
        LevelScrollBar1.Value = constants.TestGradientMagnitude;
        LevelScrollBar2.Value = constants.TestGradientWidthRelative;
        LevelScrollBar3.Value = constants.TestGradientPositionRelative;
        LevelScrollBar4.Value = constants.ZoneRadiusUm;
        LevelScrollBar5.Value = constants.ActivatedSynapsesCount;        
    }

    private void GetDataFromControls(Model01.ModelConstants constants)
    {
        constants.TestGradientAngleDegrees = (float)LevelScrollBar0.Value;
        constants.TestGradientMagnitude = (float)LevelScrollBar1.Value;
        constants.TestGradientWidthRelative = (float)LevelScrollBar2.Value;
        constants.TestGradientPositionRelative = (float)LevelScrollBar3.Value;
        constants.ZoneRadiusUm = (float)LevelScrollBar4.Value;
        constants.ActivatedSynapsesCount = (int)LevelScrollBar5.Value;        
    }

    #region Buttons Handlers    

    private void GradientInput_OnClick(object? sender, RoutedEventArgs args)
    {
        _randomInput = null;
        _rangeInput = null;

        _ = Refresh();
    }

    private void RandomInput_OnClick(object? sender, RoutedEventArgs args)
    {
        _randomInput = new float[Model01.Constants.HashLength];        
        for (int i = 0; i < 10; i += 1)
        {
            _randomInput[_random.Next(_randomInput.Length)] = 1.0f;
        }
        _rangeInput = null;

        _ = Refresh();
    }

    private void RangeInput_OnClick(object? sender, RoutedEventArgs args)
    {
        _randomInput = null;
        _rangeInput = new float[Model01.Constants.HashLength];

        _ = Refresh();
    }

    private void RangeInputLeft_OnClick(object? sender, RoutedEventArgs args)
    {
        int start = GetBitStart() - 1;
        LevelScrollBar0.Value = LevelScrollBar0.Minimum + (LevelScrollBar0.Maximum - LevelScrollBar0.Minimum) * start / Model01.Constants.HashLength;
    }

    private void RangeInputRight_OnClick(object? sender, RoutedEventArgs args)
    {
        int start = GetBitStart() + 1;
        int bitLength = GetBitLength();
        if (start > Model01.Constants.HashLength - bitLength)
            return;
        LevelScrollBar0.Value = LevelScrollBar0.Minimum + (LevelScrollBar0.Maximum - LevelScrollBar0.Minimum) * start / Model01.Constants.HashLength;
    }

    private async void RangeInputRightRight_OnClick(object? sender, RoutedEventArgs args)
    {
        int start = 0;
        int bitLength = GetBitLength();
        for (; ; )
        {
            await Task.Delay(400);
            Model01.Constants.TestGradientAngleDegrees = (float)(LevelScrollBar0.Minimum + (LevelScrollBar0.Maximum - LevelScrollBar0.Minimum) * start / Model01.Constants.HashLength);
            await Refresh();
            start += 1;
            if (start > Model01.Constants.HashLength - bitLength)
                break;
        }

        LevelScrollBar0.Value = Model01.Constants.TestGradientAngleDegrees;
    }

    #endregion

    private void Reset()
    {
        var constants = Model01.Constants;
        constants.RetinaPointDeltaPixels = 1.0f; // Для быстродействия
        constants.DetectorFieldOfViewRadiusPixels = 1.0f;
        GetDataFromControls(constants);

        _random = new Random();

        Model = new Model01(_random, new Model01.Options
        {
            OnlyCenterHyperColumn = true,
            LoadImagesSamplesFile = false,
        });

        Model!.PutPinwheel_MemoriesAndSomWeights(_random, 1);

        Model!.Cortex.CalculateSomWeightsEquivalentCortexMemories(_random);

        Model!.Create_MiniColumnDetailed(_random);
    }

    private bool RefreshTaskIsRunning
    {
        get { return _refreshTaskIsRunning; }  
        set
        {
            _refreshTaskIsRunning = value;
            if (_refreshTaskIsRunning)
                BusyImage.IsVisible = true;
            else 
                BusyImage.IsVisible = false;
        }
    }

    private bool _refreshTaskIsRunning;
    private bool _refreshTaskIsPending;

    private async Task Refresh()
    {
        Refresh_ImagesSet1();

        if (RefreshTaskIsRunning)
        {
            _refreshTaskIsPending = true;
            return;
        }

        RefreshTaskIsRunning = true;

        try
        {
            do
            {
                _refreshTaskIsPending = false; // Сбрасываем флаг перед началом

                var (cortexMemory, nearest_HyperColumnCenter_MiniColumn) = Model!.GetTestCortexMemory_SimpleDetectors(_random);
                if (_randomInput is not null)
                {
                    Array.Copy(_randomInput, cortexMemory.Hash, cortexMemory.Hash.Length);
                }
                else if (_rangeInput is not null)
                {
                    Array.Clear(_rangeInput);
                    int start = GetBitStart();
                    for (int i = 0; i < GetBitLength(); i += 1)
                    {
                        int bitIndex = start + i;
                        if (bitIndex < _rangeInput.Length)
                            _rangeInput[bitIndex] = 1.0f;
                    }
                    Array.Copy(_rangeInput, cortexMemory.Hash, cortexMemory.Hash.Length);
                }

                MainFloatVectorStripControl.Values = cortexMemory.Hash;

                Model01.ModelConstants constantsClone = JsonSerializer.Deserialize<Model01.ModelConstants>(JsonSerializer.Serialize(Model01.Constants))!;

                // Выполнение тяжелой задачи с актуальными на данный момент данными
                await Task.Run(() =>
                {
                    Model!.MiniColumnDetailedModel_Create3D_ThreadSafe(_random, cortexMemory, constantsClone);
                });

                Refresh_3D();

            } while (_refreshTaskIsPending); // Если во время работы флаг подняли, идем на новый круг
        }
        finally
        {
            RefreshTaskIsRunning = false;
        }
    }    

    private int GetBitStart()
    {
        return (int)(Model01.Constants.HashLength * Model01.Constants.TestGradientAngleDegrees / 360.0f);
    }

    private int GetBitLength()
    {
        return (int)(Model01.Constants.HashLength * Model01.Constants.TestGradientMagnitude / 1200.0f);
    }

    private async void Refresh_ImagesSet1()
    {
        if (Model is null)
            return;

        await Model!.TestMemory_FindBestForMemoryMiniColumn_SomAsync(_random, CancellationToken.None);

        ImagesSet1.MainItemsControl.ItemsSource = Model!.Get_MiniColumnDetailed_VisualizationWithDescs(
            _random);
    }

    private void Refresh_3D()
    {
        if (Model is null)
            return;

        MainModel3DControl.Data = Model!.Get_MiniColumnDetailed_Model3DScene(
            _random);
    }

    private Random _random = null!;

    private float[]? _randomInput;
    private float[]? _rangeInput;

    private CancellationTokenSource? _cancellationTokenSource;

    private Task? _curentLongRunningTask;    
}