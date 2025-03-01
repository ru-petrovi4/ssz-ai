using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Interactivity;
using Ssz.AI.Helpers;
using Ssz.AI.Models;
using Ssz.Utils;
using System;
using System.Linq;
using System.Numerics.Tensors;

namespace Ssz.AI.Views;

public partial class Model5View : UserControl
{
    public Model5View()
    {
        InitializeComponent();

        if (Design.IsDesignMode)
            return;

        _model = new Model5();

        ((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[0].Value = 0.72;
        ((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[1].Value = 0.45;
        ((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[2].Value = 0.23;

        ((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[0].Value = 0.72;
        ((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[1].Value = 0.45;
        ((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[2].Value = 0.23;

        foreach (int i in Enumerable.Range(0, _model.Constants.MiniColumnsMaxDistance + 1))
        {
            ((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[i].PropertyChanged += (s, e) => GetDataFromControls();
            ((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[i].PropertyChanged += (s, e) => GetDataFromControls();            
        }
        LevelScrollBar.ValueChanged += (s, e) => GetDataFromControls();
        LevelScrollBar2.ValueChanged += (s, e) => GetDataFromControls();
        LevelScrollBar3.ValueChanged += (s, e) => GetDataFromControls();
        LevelScrollBar4.ValueChanged += (s, e) => GetDataFromControls();
        GetDataFromControls();

        Reset();

        Refresh_ImagesSet1();    
        Refresh_ImagesSet2();
    }    

    private void Reset()
    {
        _model.ResetMemories();
        _random = new Random(1); // Pseudorandom
        _model.CurrentInputIndex = -1; // Перед первым элементом
        //_model.DoSteps_MNIST(1000, _random, initialization: true);            
    }

    private void GetDataFromControls()
    {
        foreach (int i in Enumerable.Range(0, _model.Constants.MiniColumnsMaxDistance + 1))
        {
            _model.Cortex.PositiveK[i] = (float)((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[i].Value;
            _model.Cortex.NegativeK[i] = (float)((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[i].Value;
        }
        _model.Cortex.K0 = (float)LevelScrollBar.Value;
        _model.Cortex.K1 = (float)LevelScrollBar2.Value;
        _model.Cortex.K2 = (float)LevelScrollBar3.Value;
        _model.Cortex.K3 = (float)LevelScrollBar4.Value;
    }

    private void ResetButton_OnClick(object? sender, RoutedEventArgs args)
    {
        Reset();        

        Refresh_ImagesSet2();        
    }

    private void GenerateRotator_OnClick(object? sender, RoutedEventArgs args)
    {
        Reset();

        _model.GenerateRotator();

        Refresh_ImagesSet2();
    }

    private void ProcessSamples10KButton_OnClick(object? sender, RoutedEventArgs args)
    {
        _model.DoSteps_MNIST(10000, _random, randomInitialization: false, reorderMemoriesPeriodically: true);

        Refresh_ImagesSet2();
    }

    private void ProcessSamples2000Button_OnClick(object? sender, RoutedEventArgs args)
    {
        _model.DoSteps_MNIST(2000, _random, randomInitialization: false, reorderMemoriesPeriodically: true);

        Refresh_ImagesSet2();
    }

    private void ProcessSampleButton_OnClick(object? sender, RoutedEventArgs args)
    {
        _model.DoSteps_MNIST(1, _random, randomInitialization: false, reorderMemoriesPeriodically: false);

        Refresh_ImagesSet2();
    }

    private void FloodButton_OnClick(object? sender, RoutedEventArgs args)
    {
        _model.Flood(_random);

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

    private Model5 _model = null!;

    private Random _random = null!;
}