using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Interactivity;
using Avalonia.Markup.Xaml;
using Ssz.AI.Helpers;
using Ssz.AI.Models;
using Ssz.AI.ViewModels;
using Ssz.Utils;
using System;

namespace Ssz.AI.Views;

public partial class Model5View : UserControl
{
    public Model5View()
    {
        InitializeComponent();

        //DataContext = new Model5ViewModel();

        _model5 = new Model5();

        Refresh_StackPanel1();
    }

    private void StepButton_OnClick(object? sender, RoutedEventArgs args)
    {   
        Refresh_StackPanel2();
    }

    private void VisualizeButton_OnClick(object? sender, RoutedEventArgs args)
    {
        Refresh_StackPanel3();
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
        double position = this.FindControl<ScrollBar>("PositionScrollBar")!.Value;
        double angle = this.FindControl<ScrollBar>("AngleScrollBar")!.Value;
        var images = _model5.GetImages1(position, angle);

        var panel = this.FindControl<StackPanel>("StackPanel1")!;
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
        var images = _model5.GetImages2(1);

        var panel = this.FindControl<StackPanel>("StackPanel2")!;
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

    private void Refresh_StackPanel3()
    {
        var images = _model5.GetImages3();

        var panel = this.FindControl<StackPanel>("StackPanel3")!;
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

    private Model5 _model5;
}