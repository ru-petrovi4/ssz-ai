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

        //DataContext = new Model5ViewModel();

        _model = new Model5();

        Refresh_StackPanel1();
    }

    private void StepMnistButton_OnClick(object? sender, RoutedEventArgs args)
    {
        _model.DoSteps_MNIST(1);

        //Refresh_StackPanel3();
    }

    private void StepGeneratedLineButton_OnClick(object? sender, RoutedEventArgs args)
    {
        double position = this.FindControl<ScrollBar>("PositionScrollBar")!.Value;
        double angle = this.FindControl<ScrollBar>("AngleScrollBar")!.Value;
        _model.DoStep_GeneratedLine(position, angle);

        Refresh_StackPanel1();        
    }

    private void ProcessSamplesButton_OnClick(object? sender, RoutedEventArgs args)
    {
        _model.ResetMemories();
        foreach (int i in Enumerable.Range(0, _model.Constants.MiniColumnsMaxDistance + 1))
        {
            _model.Cortex.PositiveK[i] = (float)((SlidersViewModel)PositiveSliders.DataContext!).SlidersItems[i].Value;
            _model.Cortex.NegativeK[i] = (float)((SlidersViewModel)NegativeSliders.DataContext!).SlidersItems[i].Value;
        }
        _model.Cortex.PositiveCosineSimilarity = (float)LevelScrollBar.Value;
        _model.CurrentInputIndex = -1; // Перед первым элементом
        _model.DoSteps_MNIST(5000);

        Refresh_StackPanel2();
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
        var images = _model.GetImages1(position, angle);

        PositionTextBlock.Text =
            new Any(_model.Generated_CenterXDelta).ValueAsString(false);
        AngleTextBlock.Text =
            new Any(180.0 * _model.Generated_AngleDelta / Math.PI).ValueAsString(false);
        ScalarProductTextBlock.Text =
            new Any(TensorPrimitives.CosineSimilarity(_model.DetectorsActivationHash, _model.DetectorsActivationHash0)).ValueAsString(false);

        var panel = StackPanel1;
        panel.Children.Clear();
        foreach (var image in images)
        {
            var bitmap = BitmapHelper.ConvertImageToAvaloniaBitmap(image);
            var imageControl = new Avalonia.Controls.Image
            {
                Source = bitmap,
                //Width = 150,
                //Height = 150
            };
            panel.Children.Add(imageControl);
        }
    }

    private void Refresh_StackPanel2()
    {   
        var images = _model.GetImages2();

        var panel = StackPanel2;
        panel.Children.Clear();
        foreach (var image in images)
        {
            var bitmap = BitmapHelper.ConvertImageToAvaloniaBitmap(image);
            var imageControl = new Avalonia.Controls.Image
            {
                Source = bitmap,
                //Width = 150,
                //Height = 150
            };
            panel.Children.Add(imageControl);
        }
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

    private Model5 _model;
}