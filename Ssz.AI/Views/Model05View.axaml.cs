using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Interactivity;
using Microsoft.AspNetCore.Identity;
using MsBox.Avalonia;
using Ssz.AI.Helpers;
using Ssz.AI.Models;
using Ssz.Utils;
using System;
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

        LevelScrollBar0.ValueChanged += (s, e) => GetDataFromControls(_model.Constants);
        LevelScrollBar1.ValueChanged += (s, e) => GetDataFromControls(_model.Constants);
        LevelScrollBar2.ValueChanged += (s, e) => GetDataFromControls(_model.Constants);
        LevelScrollBar3.ValueChanged += (s, e) => GetDataFromControls(_model.Constants);
        LevelScrollBar4.ValueChanged += (s, e) => GetDataFromControls(_model.Constants);
        LevelScrollBar5.ValueChanged += (s, e) => GetDataFromControls(_model.Constants);

        Reset();        

        Refresh_ImagesSet1();    
        Refresh_ImagesSet2();
    }    

    private void Reset()
    {
        var constants = new Model05.ModelConstants();
        GetDataFromControls(constants);
        _model = new Model05(constants);
        _model.ResetMemories();
        _random = new Random(3); // Pseudorandom
        _model.CurrentInputIndex = -1; // Перед первым элементом
        //_model.DoSteps_MNIST(1000, _random, initialization: true);                    
    }

    private void GetDataFromControls(IConstants constants)
    {
        //foreach (int i in Enumerable.Range(0, _model.Constants.MiniColumnsMaxDistance + 1))
        //{
        //    _model.Cortex.PositiveK[i] = (float)((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[i].Value;
        //    _model.Cortex.NegativeK[i] = (float)((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[i].Value;
        //}
        constants.K0 = (float)LevelScrollBar0.Value;
        constants.K1 = (float)LevelScrollBar1.Value;
        constants.K2 = (float)LevelScrollBar2.Value;
        constants.K3 = (float)LevelScrollBar3.Value;
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
        var lastAddedMemory = _model.Temp_ActivitiyMaxInfo?.Temp_WinnerMiniColumn?.Temp_Memory;
        if (lastAddedMemory is not null)
            _model.Temp_ActivitiyMaxInfo!.Temp_WinnerMiniColumn!.Memories.Remove(lastAddedMemory);

        _model.CurrentInputIndex -= 1;
    }

    private void GenerateRotator_OnClick(object? sender, RoutedEventArgs args)
    {
        _model.GenerateRotator(_random);

        Refresh_ImagesSet2();
    }

    private async void ProcessSamples10KButton_OnClick(object? sender, RoutedEventArgs args)
    {
        IsEnabled = false;
        await Task.Delay(50);        

        _model.CurrentInputIndex = -1;

        await _model.DoSteps_MNISTAsync(10000, _random, randomInitialization: false, reorderMemoriesPeriodically: true);

        Refresh_ImagesSet2();

        IsEnabled = true;
    }

    private async void ProcessSamples5KButton_OnClick(object? sender, RoutedEventArgs args)
    {
        IsEnabled = false;
        await Task.Delay(50);

        _model.CurrentInputIndex = -1;

        await _model.DoSteps_MNISTAsync(5000, _random, randomInitialization: false, reorderMemoriesPeriodically: true);

        Refresh_ImagesSet2();

        IsEnabled = true;
    }

    private async void ProcessSamples2000Button_OnClick(object? sender, RoutedEventArgs args)
    {
        IsEnabled = false;
        await Task.Delay(50);

        await _model.DoSteps_MNISTAsync(2000, _random, randomInitialization: false, reorderMemoriesPeriodically: false);

        Refresh_ImagesSet2();

        IsEnabled = true;
    }

    private async void ReorderMemoriesButton_OnClick(object? sender, RoutedEventArgs args)
    {
        IsEnabled = false;
        await Task.Delay(50);        

        await _model.ReorderMemoriesAsync(Int32.MaxValue, _random, async () =>
        {
            Refresh_ImagesSet2();
            await Task.Delay(50);
        });

        Refresh_ImagesSet2();

        IsEnabled = true;
    }

    private async void ProcessSampleButton_OnClick(object? sender, RoutedEventArgs args)
    {
        await _model.DoSteps_MNISTAsync(1, _random, randomInitialization: false, reorderMemoriesPeriodically: false);

        Refresh_ImagesSet2();
    }

    private void ProcessMemoryButton_OnClick(object? sender, RoutedEventArgs args)
    {
        _model.DoStep_Memory(_random);

        Refresh_ImagesSet2();
    }

    private async void FloodButton_OnClick(object? sender, RoutedEventArgs args)
    {
        var floodRadius = await DialogHelper.GetValueFromUserAsync(
                "Радиус потопа:"             // Заголовок                
            );

        _model.Flood(_random, new Any(floodRadius).ValueAsSingle(false));

        Refresh_ImagesSet2();
    }

    private void ShowTestWindowButton_OnClick(object? sender, RoutedEventArgs args)
    {
        Window testWindow = new();

        testWindow.Content = new GeneratedImages(_model);
        
        testWindow.Show((Window)Window.GetTopLevel(this)!);
    }

    //private void StepGeneratedLineButton_OnClick(object? sender, RoutedEventArgs args)
    //{
    //    double position = this.FindControl<ScrollBar>("PositionScrollBar")!.Value;
    //    double angle = this.FindControl<ScrollBar>("AngleScrollBar")!.Value;
    //    _model.DoStep_GeneratedLine(position, angle);

    //    Refresh_ImagesSet1();        
    //}    

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
        GetDataFromControls(_model.Constants);
    }

    private void Refresh_ImagesSet1()
    {
        double position = PositionScrollBar.Value;
        double angle = MathHelper.DegreesToRadians(AngleScrollBar.Value);
        ImagesSet1.MainItemsControl.ItemsSource = _model.GetImageWithDescs1(position, angle);

        PositionTextBlock.Text =
            new Any(_model.Generated_CenterXDelta).ValueAsString(false);
        AngleTextBlock.Text =
            new Any(180.0 * _model.Generated_AngleDelta / Math.PI).ValueAsString(false);
        ScalarProductTextBlock.Text =
            new Any(TensorPrimitives.CosineSimilarity(_model.DetectorsActivationHash, _model.DetectorsActivationHash0)).ValueAsString(false);
    }

    private void Refresh_ImagesSet2()
    {
        ImagesSet2.MainItemsControl.ItemsSource = _model.GetImageWithDescs2();
    }    

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

    private Model05 _model = null!;

    private Random _random = null!;
}
