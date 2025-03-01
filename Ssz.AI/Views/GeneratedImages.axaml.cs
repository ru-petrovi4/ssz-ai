using Avalonia;
using Avalonia.Controls;
using Avalonia.Markup.Xaml;
using Ssz.AI.Models;
using Ssz.Utils;
using System;
using System.Numerics.Tensors;
using System.Threading.Tasks;
using Tmds.DBus.Protocol;

namespace Ssz.AI.Views;

public partial class GeneratedImages : UserControl
{
    public GeneratedImages(Model5 model5)
    {
        InitializeComponent();

        GeneratedImage0.MagnitudeScrollBar.ValueChanged += GeneratedImage0_MagnitudeScrollBar_ValueChanged;
        GeneratedImage0.AngleScrollBar.ValueChanged += GeneratedImage0_AngleScrollBar_ValueChanged;
        GeneratedImage1.MagnitudeScrollBar.ValueChanged += GeneratedImage1_MagnitudeScrollBar_ValueChanged;
        GeneratedImage1.AngleScrollBar.ValueChanged += GeneratedImage1_AngleScrollBar_ValueChanged;
        RotatorGeneratedImage.MagnitudeScrollBar.ValueChanged += RotatorGeneratedImage_MagnitudeScrollBar_ValueChanged;
        RotatorGeneratedImage.AngleScrollBar.ValueChanged += RotatorGeneratedImage_AngleScrollBar_ValueChanged;

        _model5 = model5;

        RotatorGeneratedImage.Refresh(_model5);
    }

    private void GeneratedImage0_MagnitudeScrollBar_ValueChanged(object? sender, Avalonia.Controls.Primitives.RangeBaseValueChangedEventArgs e)
    {
        GeneratedImage0.Refresh();

        Refresh();
    }    

    private void GeneratedImage0_AngleScrollBar_ValueChanged(object? sender, Avalonia.Controls.Primitives.RangeBaseValueChangedEventArgs e)
    {
        GeneratedImage0.Refresh();

        Refresh();
    }

    private void GeneratedImage1_MagnitudeScrollBar_ValueChanged(object? sender, Avalonia.Controls.Primitives.RangeBaseValueChangedEventArgs e)
    {
        GeneratedImage1.Refresh();

        Refresh();
    }

    private void GeneratedImage1_AngleScrollBar_ValueChanged(object? sender, Avalonia.Controls.Primitives.RangeBaseValueChangedEventArgs e)
    {
        GeneratedImage1.Refresh();

        Refresh();
    }

    private void RotatorGeneratedImage_MagnitudeScrollBar_ValueChanged(object? sender, Avalonia.Controls.Primitives.RangeBaseValueChangedEventArgs e)
    {
        RotatorGeneratedImage.Refresh(_model5);
    }

    private void RotatorGeneratedImage_AngleScrollBar_ValueChanged(object? sender, Avalonia.Controls.Primitives.RangeBaseValueChangedEventArgs e)
    {
        RotatorGeneratedImage.Refresh(_model5);
    }

    private void Refresh()
    {
        Parallel.For(
                    fromInclusive: 0,
                    toExclusive: _model5.Cortex.SubArea_Detectors.Length,
                    di =>
                    {
                        var d = _model5.Cortex.SubArea_Detectors[di];
                        d.CalculateIsActivated(GeneratedImage0.GeneratedGradientMatrix);
                    });

        float[] hash0 = new float[_model5.Constants.HashLength];
        _model5.Cortex.CenterMiniColumn!.GetHash(hash0);

        Parallel.For(
                    fromInclusive: 0,
                    toExclusive: _model5.Cortex.SubArea_Detectors.Length,
                    di =>
                    {
                        var d = _model5.Cortex.SubArea_Detectors[di];
                        d.CalculateIsActivated(GeneratedImage1.GeneratedGradientMatrix);
                    });

        float[] hash1 = new float[_model5.Constants.HashLength];
        _model5.Cortex.CenterMiniColumn!.GetHash(hash1);

        CosineSimilarityTextBlock.Text = new Any(TensorPrimitives.CosineSimilarity(hash0, hash1)).ValueAsString(false, "F04");
    }

    private Model5 _model5;
}