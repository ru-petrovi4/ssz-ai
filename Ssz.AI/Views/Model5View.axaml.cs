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

        Reset();

        Refresh_StackPanel1();        
    }    

    private void Reset()
    {
        _model.ResetMemories();
        _random = new Random(1); // Pseudorandom
        _model.CurrentInputIndex = -1; // Перед первым элементом       
    }

    private void ProcessSamplesButton_OnClick(object? sender, RoutedEventArgs args)
    {
        foreach (int i in Enumerable.Range(0, _model.Constants.MiniColumnsMaxDistance + 1))
        {
            _model.Cortex.PositiveK[i] = (float)((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[i].Value;
            _model.Cortex.NegativeK[i] = (float)((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[i].Value;
        }
        _model.Cortex.PositiveCosineSimilarity = (float)LevelScrollBar.Value;             
        _model.DoSteps_MNIST(1000, _random);

        Refresh_StackPanel2();
    }

    private void ResetButton_OnClick(object? sender, RoutedEventArgs args)
    {
        Reset();

        ImagesSet2.MainItemsControl.ItemsSource = null;
    }

    private void ProcessSampleButton_OnClick(object? sender, RoutedEventArgs args)
    {        
        foreach (int i in Enumerable.Range(0, _model.Constants.MiniColumnsMaxDistance + 1))
        {
            _model.Cortex.PositiveK[i] = (float)((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[i].Value;
            _model.Cortex.NegativeK[i] = (float)((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[i].Value;
        }
        _model.Cortex.PositiveCosineSimilarity = (float)LevelScrollBar.Value;      
        _model.DoSteps_MNIST(1, _random);

        Refresh_StackPanel2();
    }    

    private void StepGeneratedLineButton_OnClick(object? sender, RoutedEventArgs args)
    {
        double position = this.FindControl<ScrollBar>("PositionScrollBar")!.Value;
        double angle = this.FindControl<ScrollBar>("AngleScrollBar")!.Value;
        _model.DoStep_GeneratedLine(position, angle);

        Refresh_StackPanel1();        
    }    

    private void PositionScrollBar_OnValueChanged(object? sender, RangeBaseValueChangedEventArgs e)
    {
        Refresh_StackPanel1();
    }

    private void AngleScrollBar_OnValueChanged(object? sender, RangeBaseValueChangedEventArgs e)
    {
        Refresh_StackPanel1();
    }

    private void Refresh_StackPanel1()
    {
        double position = PositionScrollBar.Value;
        double angle = AngleScrollBar.Value;
        ImagesSet1.MainItemsControl.ItemsSource = _model.GetImageWithDescs1(position, angle);

        PositionTextBlock.Text =
            new Any(_model.Generated_CenterXDelta).ValueAsString(false);
        AngleTextBlock.Text =
            new Any(180.0 * _model.Generated_AngleDelta / Math.PI).ValueAsString(false);
        ScalarProductTextBlock.Text =
            new Any(TensorPrimitives.CosineSimilarity(_model.DetectorsActivationHash, _model.DetectorsActivationHash0)).ValueAsString(false);
    }

    private void Refresh_StackPanel2()
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