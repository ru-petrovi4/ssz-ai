using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Interactivity;
using Avalonia.Markup.Xaml;
using Avalonia.Threading;
using Ssz.AI.Helpers;
using Ssz.AI.Models;
using Ssz.AI.ViewModels;
using Ssz.Utils;
using System;

namespace Ssz.AI.Views;

public partial class Model7View : UserControl
{
    public Model7View()
    {
        InitializeComponent();

        //DataContext = new Model5ViewModel();

        _model = new Model7();

        var timer = new DispatcherTimer
        {
            Interval = TimeSpan.FromSeconds(1) // Интервал в 1 секунду
        };

        timer.Tick += Timer_Tick;
        timer.Start();
    }

    private void Timer_Tick(object? sender, EventArgs e)
    {
        //Refresh_StackPanel2();
    }

    private void StepMnistButton_OnClick(object? sender, RoutedEventArgs args)
    {
        //_model.CollectMemories_MNIST(1);

        Refresh_StackPanel2();
    }

    private void StepGeneratedLineButton_OnClick(object? sender, RoutedEventArgs args)
    {
        double position = this.FindControl<ScrollBar>("PositionScrollBar")!.Value;
        double angle = this.FindControl<ScrollBar>("AngleScrollBar")!.Value;
        _model.DoStep_GeneratedLine(position, angle);

        Refresh_StackPanel1();        
    }

    private void StopAutoencoderFindingButton_OnClick(object? sender, RoutedEventArgs args)
    {
        _model.Temp_StopAutoencoderFinding_CancellationTokenSource.Cancel();
        _model.Temp_StopAutoencoderFinding_CancellationTokenSource = new System.Threading.CancellationTokenSource();
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
        var images = _model.GetImages1(position, angle);

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
        var images = _model.GetImages2();

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
        var images = _model.GetImages3();

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

    private Model7 _model;   
}