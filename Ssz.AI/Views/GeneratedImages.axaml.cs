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
    public GeneratedImages(Model05View model05View)
    {
        InitializeComponent();

        GeneratedImage0.MagnitudeScrollBar.ValueChanged += GeneratedImage0_MagnitudeScrollBar_ValueChanged;
        GeneratedImage0.AngleScrollBar.ValueChanged += GeneratedImage0_AngleScrollBar_ValueChanged;
        GeneratedImage1.MagnitudeScrollBar.ValueChanged += GeneratedImage1_MagnitudeScrollBar_ValueChanged;
        GeneratedImage1.AngleScrollBar.ValueChanged += GeneratedImage1_AngleScrollBar_ValueChanged;
        PinwheelGeneratedImage.MagnitudeScrollBar.ValueChanged += PinwheelGeneratedImage_MagnitudeScrollBar_ValueChanged;
        PinwheelGeneratedImage.AngleScrollBar.ValueChanged += PinwheelGeneratedImage_AngleScrollBar_ValueChanged;

        _model05View = model05View;

        PinwheelGeneratedImage.Refresh(model05View.Model);
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

    private void PinwheelGeneratedImage_MagnitudeScrollBar_ValueChanged(object? sender, Avalonia.Controls.Primitives.RangeBaseValueChangedEventArgs e)
    {
        PinwheelGeneratedImage.Refresh(_model05View.Model);
    }

    private void PinwheelGeneratedImage_AngleScrollBar_ValueChanged(object? sender, Avalonia.Controls.Primitives.RangeBaseValueChangedEventArgs e)
    {
        PinwheelGeneratedImage.Refresh(_model05View.Model);
    }

    private void Refresh()
    {
        var model = _model05View.Model;
        Parallel.For(
                    fromInclusive: 0,
                    toExclusive: model.Cortex.SubArea_Detectors.Length,
                    di =>
                    {
                        var d = model.Cortex.SubArea_Detectors[di];
                        d.CalculateIsActivated(model.Retina, GeneratedImage0.GeneratedGradientMatrix, model.Cortex.Constants);
                    });

        float[] hash0 = new float[model.Constants.HashLength];
        model.Cortex.CenterMiniColumn!.GetHash(hash0);

        Parallel.For(
                    fromInclusive: 0,
                    toExclusive: model.Cortex.SubArea_Detectors.Length,
                    di =>
                    {
                        var d = model.Cortex.SubArea_Detectors[di];
                        d.CalculateIsActivated(model.Retina, GeneratedImage1.GeneratedGradientMatrix, model.Cortex.Constants);
                    });

        float[] hash1 = new float[model.Constants.HashLength];
        model.Cortex.CenterMiniColumn!.GetHash(hash1);

        CosineSimilarityTextBlock.Text = new Any(TensorPrimitives.CosineSimilarity(hash0, hash1)).ValueAsString(false, "F04");
    }

    private Model05View _model05View;
}