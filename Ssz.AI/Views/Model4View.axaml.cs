using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Markup.Xaml;
using Ssz.AI.Helpers;
using Ssz.AI.Models;
using Ssz.AI.ViewModels;
using Ssz.Utils;
using System;

namespace Ssz.AI.Views;

public partial class Model4View : UserControl
{
    public Model4View()
    {
        InitializeComponent();

        //DataContext = new Model4ViewModel();

        _model4 = new Model4();

        Refresh();
    }

    private void PositionScrollBar_OnValueChanged(object? sender, RangeBaseValueChangedEventArgs e)
    {
        Refresh();
    }    

    private void AngleScrollBar_OnValueChanged(object? sender, RangeBaseValueChangedEventArgs e)
    {
        Refresh();
    }

    private void Refresh()
    {
        double position = this.FindControl<ScrollBar>("PositionScrollBar")!.Value;
        double angle = this.FindControl<ScrollBar>("AngleScrollBar")!.Value;
        var images = _model4.GetImages(position, angle);

        this.FindControl<TextBlock>("PositionTextBlock")!.Text = 
            new Any(_model4.CenterXDelta).ValueAsString(false);
        this.FindControl<TextBlock>("AngleTextBlock")!.Text = 
            new Any(180.0 * _model4.AngleDelta / Math.PI).ValueAsString(false);
        this.FindControl<TextBlock>("ScalarProductTextBlock")!.Text = 
            new Any(_model4.DetectorsActivationScalarProduct / _model4.DetectorsActivationScalarProduct0).ValueAsString(false);

        var panel = this.FindControl<StackPanel>("MainStackPanel")!;
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

    private Model4 _model4;
}