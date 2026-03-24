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
            Refresh();
        };
        LevelScrollBar1.ValueChanged += (s, e) =>
        {
            GetDataFromControls(constants);
            Refresh();
        };
        LevelScrollBar2.ValueChanged += (s, e) =>
        {
            GetDataFromControls(constants);
            Refresh();
        };
        LevelScrollBar3.ValueChanged += (s, e) =>
        {
            GetDataFromControls(constants);
            Refresh();
        };
        LevelScrollBar4.ValueChanged += (s, e) =>
        {
            GetDataFromControls(constants);
            Refresh();
        };
        LevelScrollBar5.ValueChanged += (s, e) =>
        {
            GetDataFromControls(constants);
            Refresh();
        };

        Reset();
        Refresh();
        //if (true)
        //{
        //    Reset();
        //    Refresh_ImagesSet();
        //}
        //else
        //{
        //    var t = Script2Async();
        //}
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

    //                Model.Logger.LogInformation("StartProcessSomIdealN Started.");

    //                await Model.ProcessSomNAsync(epochsCount: null, _random, CancellationToken.None, () =>
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
    //                Model.Logger.LogInformation("StartProcessSomIdealN Cancelled.");
    //            }
    //            Model.Logger.LogInformation("StartProcessSomIdealN Finished.");
    //        });

    //        var imageWithDesc = (ImageWithDesc)(Model.GetImageWithDescs(
    //            _random,
    //            ColorLowScrollBar.Value,
    //            ColorHighScrollBar.Value)[5]);

    //        imageWithDesc.Image!.Save(Path.Combine("Data", "Script2", FileSystemHelper.ReplaceInvalidChars($"{new Any(startDateTime).ValueAsString(false)}_{new Any(it).ValueAsString(false, "F02")}.png")));
    //    }
    //}

    public Model01 Model = null!;

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

    private void MiniColumnDetailedModel_PrepareCreate3D_OnClick(object? sender, RoutedEventArgs args)
    {
        Model.MiniColumnDetailedModel_PrepareCreate3D(_random);
    }

    private void MiniColumnDetailedModel_Create3D_OnClick(object? sender, RoutedEventArgs args)
    {        
        Model.MiniColumnDetailedModel_Create3D(_random);

        Refresh_ImagesSet2();
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

        Model.PutMemories_Pinwheel(_random, 1);

        Model.Cortex.CalculateSomCortexMemories(_random);
    }

    private void Refresh()
    {
        //Model.MiniColumnDetailedModel_Create3D(_random);

        Refresh_ImagesSet1();
        //Refresh_ImagesSet2();
    }

    private async void Refresh_ImagesSet1()
    {
        await Model.CalculateTestMemoryWithSomAsync(_random, CancellationToken.None);

        ImagesSet1.MainItemsControl.ItemsSource = Model.GetImageWithDescs_MiniColumnDetailed1(
            _random);
    }

    private void Refresh_ImagesSet2()
    {
        ImagesSet2.MainItemsControl.ItemsSource = Model.GetImageWithDescs_MiniColumnDetailed2(
            _random);
    }

    private Random _random = null!;

    private CancellationTokenSource? _cancellationTokenSource;

    private Task? _curentLongRunningTask;    
}