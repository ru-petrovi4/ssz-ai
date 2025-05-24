using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Interactivity;
using Microsoft.AspNetCore.Identity;
using MsBox.Avalonia;
using Ssz.AI.Helpers;
using Ssz.AI.Models;
using Ssz.Utils;
using System;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using System.Threading.Tasks;
using static Ssz.AI.Models.Cortex;

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

        LevelScrollBar0.ValueChanged += (s, e) => GetDataFromControls(Model.Constants);
        LevelScrollBar1.ValueChanged += (s, e) => GetDataFromControls(Model.Constants);
        LevelScrollBar2.ValueChanged += (s, e) => GetDataFromControls(Model.Constants);
        LevelScrollBar30.ValueChanged += (s, e) => GetDataFromControls(Model.Constants);
        LevelScrollBar31.ValueChanged += (s, e) => GetDataFromControls(Model.Constants);
        LevelScrollBar32.ValueChanged += (s, e) => GetDataFromControls(Model.Constants);
        LevelScrollBar4.ValueChanged += (s, e) => GetDataFromControls(Model.Constants);
        LevelScrollBar5.ValueChanged += (s, e) => GetDataFromControls(Model.Constants);

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
        _random = new Random(5); // Pseudorandom
        Model.CurrentInputIndex = -1; // Перед первым элементом              
    }

    private void SetDataToControls(Model05.ModelConstants constants)
    {
        LevelScrollBar0.Value = constants.K0;
        LevelScrollBar1.Value = constants.K1;
        LevelScrollBar2.Value = constants.K2;
        LevelScrollBar30.Value = constants.K3[0];
        LevelScrollBar31.Value = constants.K3[1];
        LevelScrollBar32.Value = constants.K3[2];
        LevelScrollBar4.Value = constants.K4;
        LevelScrollBar5.Value = constants.K5;
        SuperactivityThreshold.IsChecked = constants.SuperactivityThreshold;
    }

    private void GetDataFromControls(IConstants constants)
    {        
        constants.K0 = (float)LevelScrollBar0.Value;
        constants.K1 = (float)LevelScrollBar1.Value;
        constants.K2 = (float)LevelScrollBar2.Value;
        constants.K3[0] = (float)LevelScrollBar30.Value;
        constants.K3[1] = (float)LevelScrollBar31.Value;
        constants.K3[2] = (float)LevelScrollBar32.Value;
        constants.K4 = (float)LevelScrollBar4.Value;
        constants.K5 = (float)LevelScrollBar5.Value;
        constants.SuperactivityThreshold = SuperactivityThreshold.IsChecked == true;
    }

    private void ResetButton_OnClick(object? sender, RoutedEventArgs args)
    {
        Reset();        

        Refresh_ImagesSet2();        
    }

    private void Back1Button_OnClick(object? sender, RoutedEventArgs args)
    {
        var lastAddedMemory = Model.Temp_ActivitiyMaxInfo?.Temp_WinnerMiniColumn?.Temp_Memory;
        if (lastAddedMemory is not null)
            Model.Temp_ActivitiyMaxInfo!.Temp_WinnerMiniColumn!.Memories.Remove(lastAddedMemory);

        Model.CurrentInputIndex -= 1;
    }

    private void GenerateRotator_OnClick(object? sender, RoutedEventArgs args)
    {
        Model.GenerateRotator(_random);

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

    private async void DoScript_OnClick(object? sender, RoutedEventArgs args)
    {
        IsEnabled = false;

        var constants = new Model05.ModelConstants();
        GetDataFromControls(constants);

        Directory.CreateDirectory($"Data\\Script");

        int interationN = 0;
        //for (float k31 = 0.14f; k31 < 0.17f; k31 += 0.002f)
        //    for (float k32 = 0.04f; k32 < 0.07f; k32 += 0.002f)
        for (float v = 1.0f; v < 2.0f; v += 0.02f)
        {
                interationN += 1;

            //constants.K3[1] = k31;
            //constants.K3[2] = k32;
            constants.K5 = v;
           
                Model = new Model05(constants);
                Model.CurrentInputIndex = -1; // Перед первым элементом 

                await Model.DoSteps_MNISTAsync(5000, _random, randomInitialization: false, reorderMemoriesPeriodically: true);

                Model.Flood(_random, 2.5f);

                Refresh_ImagesSet2();

                var memoriesColorImage = BitmapHelper.GetSubBitmap(
                    Visualisation.GetBitmapFromMiniColumsMemoriesColor(Model.Cortex),
                    Model.Cortex.MiniColumns.Dimensions[0] / 2,
                    Model.Cortex.MiniColumns.Dimensions[1] / 2,
                    Model.Cortex.SubArea_MiniColumns_Radius + 2);

            //Разделяем на целую и дробную части
            int whole = (int)v;
            double fractional = v - whole;
            memoriesColorImage.Save($"Data\\Script\\Result {whole:D3}.{fractional.ToString("F3").Split('.')[1]}.png", ImageFormat.Png);

            //int whole1 = (int)k31;
            //double fractional1 = k31 - whole1;                
            //int whole2 = (int)k32;
            //double fractional2 = k32 - whole1;
            //memoriesColorImage.Save($"Data\\Script\\Result {whole1:D3}.{fractional1.ToString("F3").Split('.')[1]} {whole2:D3}.{fractional2.ToString("F3").Split('.')[1]}.png", ImageFormat.Png);


            await Task.Delay(50);            
            }

        IsEnabled = true;               
    }

    private Random _random = null!;
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