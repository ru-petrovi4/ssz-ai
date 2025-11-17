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
using static Ssz.AI.Models.Cortex_Simplified;

namespace Ssz.AI.Views;

public partial class Model05View : UserControl
{
    public Model05View()
    {
        InitializeComponent();

        if (Design.IsDesignMode)
            return;
        
        var constants = new Model05.ModelConstants();
        SetDataToControls(constants);

        LevelScrollBar0.ValueChanged += (s, e) => GetDataFromControls(constants);
        LevelScrollBar1.ValueChanged += (s, e) => GetDataFromControls(constants);
        LevelScrollBar2.ValueChanged += (s, e) => GetDataFromControls(constants);
        LevelScrollBar30.ValueChanged += (s, e) => GetDataFromControls(constants);
        LevelScrollBar31.ValueChanged += (s, e) => GetDataFromControls(constants);
        //LevelScrollBar32.ValueChanged += (s, e) => GetDataFromControls(constants);
        LevelScrollBar4.ValueChanged += (s, e) => GetDataFromControls(constants);
        LevelScrollBar5.ValueChanged += (s, e) => GetDataFromControls(constants);

        Reset();        

        Refresh_ImagesSet1();    
        Refresh_ImagesSet2();
    }

    public Model05 Model = null!;

    private void Reset()
    {
        var constants = new Model05.ModelConstants();
        GetDataFromControls(constants);
        Model = new Model05(constants);
        InitializePseudoRandom(); // Pseudorandom
        Model.CurrentInputIndex = -1; // Перед первым элементом              
    }

    private void InitializePseudoRandom()
    {
        _random = new Random(10);
    }

    private void SetDataToControls(Model05.ModelConstants constants)
    {
        LevelScrollBar0.Value = constants.K0;
        LevelScrollBar1.Value = constants.K1;
        LevelScrollBar2.Value = constants.K2;
        LevelScrollBar30.Value = constants.K3[0];
        LevelScrollBar31.Value = constants.K3[1];        
        LevelScrollBar4.Value = constants.K4;
        LevelScrollBar5.Value = constants.K5;
        SuperactivityThreshold.IsChecked = constants.SuperactivityThreshold;

        ((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[0].Value = constants.PositiveK[0];
        ((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[1].Value = constants.PositiveK[1];
        ((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[2].Value = constants.PositiveK[2];
        ((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[3].Value = constants.PositiveK[3];

        ((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[0].Value = constants.NegativeK[0];
        ((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[1].Value = constants.NegativeK[1];
        ((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[2].Value = constants.NegativeK[2];
        ((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[3].Value = constants.NegativeK[3];
    }

    private void GetDataFromControls(IConstants constants)
    {        
        constants.K0 = (float)LevelScrollBar0.Value;
        constants.K1 = (float)LevelScrollBar1.Value;
        constants.K2 = (float)LevelScrollBar2.Value;
        constants.K3[0] = (float)LevelScrollBar30.Value;
        constants.K3[1] = (float)LevelScrollBar31.Value;        
        constants.K4 = (float)LevelScrollBar4.Value;
        constants.K5 = (float)LevelScrollBar5.Value;
        constants.SuperactivityThreshold = SuperactivityThreshold.IsChecked == true;

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

        Refresh_ImagesSet2();        
    }

    private void Back1Button_OnClick(object? sender, RoutedEventArgs args)
    {
        var lastAddedMemory = Model.Temp_ActivitiyMaxInfo?.SelectedSuperActivityMax_MiniColumn?.Temp_Memory;
        if (lastAddedMemory is not null)
            Model.Temp_ActivitiyMaxInfo!.SelectedSuperActivityMax_MiniColumn!.Memories.Remove(lastAddedMemory);

        Model.CurrentInputIndex -= 1;
    }

    private void GeneratePinwheel_OnClick(object? sender, RoutedEventArgs args)
    {
        Model.GeneratePinwheel(_random);

        Refresh_ImagesSet2();
    }

    private void GeneratePinwheel2_OnClick(object? sender, RoutedEventArgs args)
    {
        Model.GeneratePinwheel2(_random);

        Refresh_ImagesSet2();
    }

    private async void ProcessSamples10KButton_OnClick(object? sender, RoutedEventArgs args)
    {
        IsEnabled = false;
        await Task.Delay(50);        

        Model.CurrentInputIndex = -1;

        await Model.DoSteps_MNISTAsync(10000, _random, randomInitialization: false, reorderMemoriesPeriodically: true);

        Refresh_ImagesSet2();

        IsEnabled = true;
    }

    private async void ProcessSamples5KButton_OnClick(object? sender, RoutedEventArgs args)
    {
        IsEnabled = false;
        await Task.Delay(50);

        Model.CurrentInputIndex = -1;

        await Model.DoSteps_MNISTAsync(5000, _random, randomInitialization: false, reorderMemoriesPeriodically: true);

        Refresh_ImagesSet2();

        IsEnabled = true;
    }

    private async void ProcessSamples2000Button_OnClick(object? sender, RoutedEventArgs args)
    {
        IsEnabled = false;
        await Task.Delay(50);

        await Model.DoSteps_MNISTAsync(2000, _random, randomInitialization: false, reorderMemoriesPeriodically: false);

        Refresh_ImagesSet2();

        IsEnabled = true;
    }

    private async void ReorderMemoriesButton_OnClick(object? sender, RoutedEventArgs args)
    {
        IsEnabled = false;
        await Task.Delay(50);        

        await Model.ReorderMemoriesAsync(Int32.MaxValue, _random, async () =>
        {
            Refresh_ImagesSet2();
            await Task.Delay(50);
        });

        Refresh_ImagesSet2();

        IsEnabled = true;
    }

    private async void ProcessSampleButton_OnClick(object? sender, RoutedEventArgs args)
    {
        await Model.DoSteps_MNISTAsync(1, _random, randomInitialization: false, reorderMemoriesPeriodically: false);

        Refresh_ImagesSet2();
    }

    private void ProcessMemoryButton_OnClick(object? sender, RoutedEventArgs args)
    {
        Model.DoStep_Memory(_random);

        Refresh_ImagesSet2();
    }

    private async void FloodButton_OnClick(object? sender, RoutedEventArgs args)
    {
        var floodRadius = await DialogHelper.GetValueFromUserAsync(
                "Радиус потопа:"             // Заголовок                
            );

        Model.Flood(_random, new Any(floodRadius).ValueAsSingle(false));

        Refresh_ImagesSet2();
    }

    private void ShowTestWindowButton_OnClick(object? sender, RoutedEventArgs args)
    {
        Window testWindow = new();

        testWindow.Content = new GeneratedImages(this);
        
        testWindow.Show((Window)Window.GetTopLevel(this)!);
    }       

    private void PositionScrollBar_OnValueChanged(object? sender, RangeBaseValueChangedEventArgs e)
    {
        Refresh_ImagesSet1();
    }    

    private void AngleScrollBar_OnValueChanged(object? sender, RangeBaseValueChangedEventArgs e)
    {
        Refresh_ImagesSet1();
    }

    private void SuperactivityThreshold_OnClick(object? sender, RoutedEventArgs args)
    {
        GetDataFromControls(Model.Constants);
    }

    private void Refresh_ImagesSet1()
    {
        double position = PositionScrollBar.Value;
        double angle = MathHelper.DegreesToRadians((float)AngleScrollBar.Value);
        ImagesSet1.MainItemsControl.ItemsSource = Model.GetImageWithDescs1(position, angle);

        PositionTextBlock.Text =
            new Any(Model.Generated_CenterXDelta).ValueAsString(false);
        AngleTextBlock.Text =
            new Any(180.0 * Model.Generated_AngleDelta / Math.PI).ValueAsString(false);
        ScalarProductTextBlock.Text =
            new Any(TensorPrimitives.CosineSimilarity(Model.DetectorsActivationHash, Model.DetectorsActivationHash0)).ValueAsString(false);
    }

    private void Refresh_ImagesSet2()
    {
        ImagesSet2.MainItemsControl.ItemsSource = Model.GetImageWithDescs2();
    }

    //private async void DoScript_OnClick(object? sender, RoutedEventArgs args)
    //{
    //    IsEnabled = false;

    //    var constants = new Model05.ModelConstants();
    //    GetDataFromControls(constants);

    //    Directory.CreateDirectory($"Data\\Script");

    //    int interationN = 0;

    //    for (float v = 1.7f; v < 2.0f; v += 0.005f)
    //    {
    //        interationN += 1;

    //        constants.K5 = v;

    //        Model = new Model05(constants);
    //        InitializePseudoRandom();
    //        Model.CurrentInputIndex = -1; // Перед первым элементом 

    //        await Model.DoSteps_MNISTAsync(5000, _random, randomInitialization: false, reorderMemoriesPeriodically: true);

    //        await Model.ReorderMemoriesAsync(Int32.MaxValue, _random, async () =>
    //        {
    //        });

    //        Model.Flood(_random, 5.0f);

    //        Refresh_ImagesSet2();

    //        var memoriesColorImage = BitmapHelper.GetSubBitmap(
    //            Visualisation.GetBitmapFromMiniColumsMemoriesColor(Model.Cortex),
    //            Model.Cortex.MiniColumns.Dimensions[0] / 2,
    //            Model.Cortex.MiniColumns.Dimensions[1] / 2,
    //            Model.Cortex.SubArea_MiniColumns_Radius + 2);

    //        // Разделяем на целую и дробную части
    //        int whole = (int)v;
    //        double fractional = v - whole;
    //        memoriesColorImage.Save($"Data\\Script\\Result {whole:D3}.{fractional.ToString("F3").Split('.')[1]}.png", ImageFormat.Png);

    //        //int whole1 = (int)k31;
    //        //double fractional1 = k31 - whole1;                
    //        //int whole2 = (int)k32;
    //        //double fractional2 = k32 - whole1;
    //        //memoriesColorImage.Save($"Data\\Script\\Result {whole1:D3}.{fractional1.ToString("F3").Split('.')[1]} {whole2:D3}.{fractional2.ToString("F3").Split('.')[1]}.png", ImageFormat.Png);

    //        await Task.Delay(1);
    //    }

    //    IsEnabled = true;
    //}

    private async void DoScript_OnClick(object? sender, RoutedEventArgs args)
    {
        IsEnabled = false;

        var constants = new Model05.ModelConstants();
        GetDataFromControls(constants);

        Directory.CreateDirectory($"Data\\Script");

        BestPinwheelSettings bestPinwheelSettings = new();

        int interationN = 0;
        for (float pk1 = 0.11f; pk1 < 0.15f; pk1 += 0.01f)
            for (float pk2 = 0.005f; pk2 < pk1; pk2 += 0.005f)
                for (float pk3 = 0.000f; pk3 < pk2; pk3 += 1.01f)
                    for (float nk1 = pk1; nk1 <= pk1; nk1 += 1.01f)
                        for (float nk2 = pk2 + 0.005f; nk2 < nk1; nk2 += 0.005f)
                            for (float nk3 = 0.000f; nk3 < nk2; nk3 += 1.01f)
                            {
                                interationN += 1;

                                constants.PositiveK[1] = pk1;
                                constants.PositiveK[2] = pk2;
                                constants.PositiveK[3] = pk3;
                                constants.NegativeK[1] = nk1;
                                constants.NegativeK[2] = nk2;
                                constants.NegativeK[3] = nk3;

                                Model = new Model05(constants);
                                InitializePseudoRandom();
                                Model.CurrentInputIndex = -1; // Перед первым элементом 

                                await Model.DoSteps_MNISTAsync(5000, _random, randomInitialization: false, reorderMemoriesPeriodically: true);

                                await Model.ReorderMemoriesAsync(10, _random, async () =>
                                {
                                });

                                float pinwheellIndex = Model.GetPinwheelIndex();
                                if (pinwheellIndex > bestPinwheelSettings.MaxPinwheelIndex)
                                {
                                    bestPinwheelSettings.MaxPinwheelIndex = pinwheellIndex;
                                    bestPinwheelSettings.Pk1 = pk1;
                                    bestPinwheelSettings.Pk2 = pk2;
                                    bestPinwheelSettings.Pk3 = pk3;
                                    bestPinwheelSettings.Nk1 = nk1;
                                    bestPinwheelSettings.Nk2 = nk2;
                                    bestPinwheelSettings.Nk3 = nk3;
                                }

                                Model.UserFriendlyLogger.LogInformation(CsvHelper.FormatForCsv(
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
                                    pk3,
                                    nk1,
                                    nk2,
                                    nk3 ]));
                            }


        IsEnabled = true;
    }

    private Random _random = null!;

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

//((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[0].Value = 1.00;
//((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[1].Value = 0.16;
//((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[2].Value = 0.02;
//((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[0].Value = 0.0;
//((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[1].Value = 0.0;
//((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[2].Value = 0.0;
//foreach (int i in Enumerable.Range(0, _model.Constants.MiniColumnsMaxDistance + 1))
//{
//    ((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[i].PropertyChanged += (s, e) => GetDataFromControls();
//    ((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[i].PropertyChanged += (s, e) => GetDataFromControls();            
//}        


//private void Refresh_StackPanel3()
//{
//    var images = _model.GetImages3();

//    var panel = StackPanel3;
//    panel.Children.Clear();
//    foreach (var image in images)
//    {
//        var bitmap = BitmapHelper.ConvertImageToAvaloniaBitmap(image);
//        var imageControl = new Avalonia.Controls.Image
//        {
//            Source = bitmap,
//            //Width = 150,
//            //Height = 150
//        };
//        panel.Children.Add(imageControl);
//    }
//}

//private void StepGeneratedLineButton_OnClick(object? sender, RoutedEventArgs args)
//{
//    double position = this.FindControl<ScrollBar>("PositionScrollBar")!.Value;
//    double angle = this.FindControl<ScrollBar>("AngleScrollBar")!.Value;
//    _model.DoStep_GeneratedLine(position, angle);

//    Refresh_ImagesSet1();        
//} 


//foreach (int i in Enumerable.Range(0, _model.Constants.MiniColumnsMaxDistance + 1))
//{
//    _model.Cortex.PositiveK[i] = (float)((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[i].Value;
//    _model.Cortex.NegativeK[i] = (float)((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[i].Value;
//}