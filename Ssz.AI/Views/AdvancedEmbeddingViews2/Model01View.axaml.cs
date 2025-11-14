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
using Ssz.AI.Models.AdvancedEmbeddingModel2;

namespace Ssz.AI.Views.AdvancedEmbeddingViews2;

public partial class Model01View : UserControl
{
    public const string FileName_Cortex = "AdvancedEmbedding2_Cortex.bin";

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

        Reset();

        Refresh_ImagesSet1();
    }

    public Model01 Model = null!;

    private void SetDataToControls(Model01.ModelConstants constants)
    {
        LevelScrollBar0.Value = constants.K0;
        LevelScrollBar1.Value = constants.K1;
        LevelScrollBar2.Value = constants.K2;        
        LevelScrollBar3.Value = constants.K3;
        LevelScrollBar4.Value = constants.K4;        

        ((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[0].Value = constants.PositiveK[0];
        ((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[1].Value = constants.PositiveK[1];
        ((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[2].Value = constants.PositiveK[2];
        ((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[3].Value = constants.PositiveK[3];

        ((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[0].Value = constants.NegativeK[0];
        ((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[1].Value = constants.NegativeK[1];
        ((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[2].Value = constants.NegativeK[2];
        ((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[3].Value = constants.NegativeK[3];
    }

    private void GetDataFromControls(Model01.ModelConstants constants)
    {
        constants.K0 = (float)LevelScrollBar0.Value;
        constants.K1 = (float)LevelScrollBar1.Value;
        constants.K2 = (float)LevelScrollBar2.Value;        
        constants.K3 = (float)LevelScrollBar3.Value;
        constants.K4 = (float)LevelScrollBar4.Value;        

        constants.PositiveK[0] = (float)((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[0].Value;
        constants.PositiveK[1] = (float)((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[1].Value;
        constants.PositiveK[2] = (float)((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[2].Value;
        constants.PositiveK[3] = (float)((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[3].Value;

        constants.NegativeK[0] = (float)((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[0].Value;
        constants.NegativeK[1] = (float)((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[1].Value;
        constants.NegativeK[2] = (float)((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[2].Value;
        constants.NegativeK[3] = (float)((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[3].Value;
    }

    private void ResetButton_OnClick(object? sender, RoutedEventArgs args)
    {
        Reset();

        Refresh_ImagesSet1();
    }

    private void ProcessSampleButton_OnClick(object? sender, RoutedEventArgs args)
    {
        Model.CalculateCortexMemories(1, _random);

        Refresh_ImagesSet1();
    }

    private void ProcessSamples2000Button_OnClick(object? sender, RoutedEventArgs args)
    {
        Model.CalculateCortexMemories(2000, _random);

        Refresh_ImagesSet1();
    }

    private void ProcessSamples5KButton_OnClick(object? sender, RoutedEventArgs args)
    {
        Model.CalculateCortexMemories(5000, _random);

        Refresh_ImagesSet1();
    }

    private void ProcessSamples10KButton_OnClick(object? sender, RoutedEventArgs args)
    {
        Model.CalculateCortexMemories(10000, _random);

        Refresh_ImagesSet1();
    }

    private void ReorderMemories1EpochButton_OnClick(object? sender, RoutedEventArgs args)
    {
        Model.ReorderMemories(1, _random, async () =>
        {
            Refresh_ImagesSet1();
            await Task.Delay(50);
        });

        Refresh_ImagesSet1();
    }

    private void ReorderMemoriesButton_OnClick(object? sender, RoutedEventArgs args)
    {
        Model.ReorderMemories(100, _random, async () =>
        {
            Refresh_ImagesSet1();
            await Task.Delay(50);
        });

        Refresh_ImagesSet1();
    }

    private void ProcessWordButton_OnClick(object? sender, RoutedEventArgs args)
    {
        Model.CalculateWords(1, _random);

        Refresh_ImagesSet1();
    }

    private async void DoScriptButton0_OnClick(object? sender, RoutedEventArgs args)
    {
        await Task.Delay(50);

        for (; ; )
        {
            bool finished = Model.CalculateCortexMemories(2000, _random);

            Model.ReorderMemories(3, _random);

            if (finished)
                break;
        }

        Model.ReorderMemories(30, _random);
        Helpers.SerializationHelper.SaveToFile(FileName_Cortex, Model.Cortex, null, Model.LoggersSet.UserFriendlyLogger);

        Refresh_ImagesSet1();
    }

    private async void DoScriptButton1_OnClick(object? sender, RoutedEventArgs args)
    {
        await Task.Delay(50);
        
        Helpers.SerializationHelper.LoadFromFileIfExists(FileName_Cortex, Model.Cortex, null, Model.LoggersSet.UserFriendlyLogger);
        Model.Cortex.Prepare();

        Model.Cortex.CalculateWords_DiscreteOptimizedVectors(_random);
        Helpers.SerializationHelper.SaveToFile(FileName_Cortex, Model.Cortex, null, Model.LoggersSet.UserFriendlyLogger);

        Refresh_ImagesSet1();
    }

    private async void DoScriptButton2_OnClick(object? sender, RoutedEventArgs args)
    {
        await Task.Delay(50);

        Helpers.SerializationHelper.LoadFromFileIfExists(FileName_Cortex, Model.Cortex, null, Model.LoggersSet.UserFriendlyLogger);

        Model.Cortex.CalculateWords_DiscreteOptimizedVectors_Metrics(_random, Model.LoggersSet.UserFriendlyLogger);
    }

    private void Reset()
    {
        var constants = Model01.Constants;
        GetDataFromControls(constants);

        _random = new Random(41);

        Model = new Model01();
        Model.PrepareCalculate(_random);
    }

    private void Refresh_ImagesSet1()
    {
        ImagesSet1_TextBlock.Text = Model.GetCurrentDesc();
        ImagesSet1.MainItemsControl.ItemsSource = Model.GetImageWithDescs1();
    }

    private Random _random = null!;
}