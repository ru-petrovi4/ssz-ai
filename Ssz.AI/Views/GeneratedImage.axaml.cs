using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Markup.Xaml;
using Ssz.AI.Helpers;
using Ssz.AI.Models;
using Ssz.AI.ViewModels;
using System;

namespace Ssz.AI.Views;

public partial class GeneratedImage : UserControl
{
    public GeneratedImage()
    {
        InitializeComponent();

        Refresh();
    }

    public DenseMatrix<GradientInPoint> GeneratedGradientMatrix { get; private set; } = null!;

    public void Refresh()
    {
        int width = MNISTHelper.MNISTImageWidthPixels;
        int height = MNISTHelper.MNISTImageHeightPixels;

        GeneratedGradientMatrix = new DenseMatrix<GradientInPoint>(width, height);

        // Вычисляем магнитуду и угол градиента
        double magnitude = MagnitudeScrollBar.Value;
        // [-pi, pi]
        double angle = MathHelper.DegreesToRadians(AngleScrollBar.Value); // Угол в радианах

        int gradX = (int)(Math.Cos(angle) * magnitude);
        int gradY = (int)(Math.Sin(angle) * magnitude);

        GradientInPoint gradientInPoint = new()
        {
            GradX = gradX,
            GradY = gradY,
            Magnitude = magnitude,
            Angle = angle,
        };

        for (int y = 1; y < height - 1; y += 1)
        {
            for (int x = 1; x < width - 1; x += 1)
            {
                GeneratedGradientMatrix[x, y] = gradientInPoint;
            }
        }

        var gradientBitmap0 = Visualisation.GetGradientBigBitmap(GeneratedGradientMatrix);
        var r = Visualisation.GetGeneratedLine_GradientMatrix(width, height, 0, angle / (2.0 * Math.PI));

        ImageWithDesc[] imageWithDescs = [
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(gradientBitmap0),
                    Desc = @"Полная картина градиентов" },
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(r.Item2),
                    Desc = @"Полная картина градиентов" },
                ];

        ImagesSet1.MainItemsControl.ItemsSource = imageWithDescs;
    }
}