using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Markup.Xaml;
using Ssz.AI.Helpers;
using Ssz.AI.Models;
using Ssz.AI.ViewModels;
using System;

namespace Ssz.AI.Views;

public partial class RotatorGeneratedImage : UserControl
{
    public RotatorGeneratedImage()
    {
        InitializeComponent();
    }

    public DenseMatrix<GradientInPoint> GeneratedGradientMatrix { get; private set; } = null!;

    public void Refresh(Model05 model)
    {
        int width = MNISTHelper.MNISTImageWidthPixels;
        int height = MNISTHelper.MNISTImageHeightPixels;

        GeneratedGradientMatrix = new DenseMatrix<GradientInPoint>(width, height);

        // Вычисляем магнитуду и угол градиента
        double magnitude = MagnitudeScrollBar.Value;
        // [-pi, pi]
        double angle = MathHelper.DegreesToRadians((float)AngleScrollBar.Value); // Угол в радианах

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

        ActivitiyMaxInfo activitiyMaxInfo = new();

        model.CalculateDetectorsAndActivityAndSuperActivity(GeneratedGradientMatrix, activitiyMaxInfo);

        var activityColorImage = BitmapHelper.GetSubBitmap(
                Visualisation.GetBitmapFromMiniColums_ActivityColor(model.Cortex),
                model.Cortex.MiniColumns.Dimensions[0] / 2,
                model.Cortex.MiniColumns.Dimensions[1] / 2,
                model.Cortex.SubArea_MiniColumns_Radius + 2);

        var superActivityColorImage = BitmapHelper.GetSubBitmap(
            Visualisation.GetBitmapFromMiniColums_SuperActivityColor(model.Cortex, activitiyMaxInfo),
            model.Cortex.MiniColumns.Dimensions[0] / 2,
            model.Cortex.MiniColumns.Dimensions[1] / 2,
            model.Cortex.SubArea_MiniColumns_Radius + 2);        

        var memoriesColorImage = BitmapHelper.GetSubBitmap(
            Visualisation.GetBitmapFromMiniColumsMemoriesColor(model.Cortex),
            model.Cortex.MiniColumns.Dimensions[0] / 2,
            model.Cortex.MiniColumns.Dimensions[1] / 2,
            model.Cortex.SubArea_MiniColumns_Radius + 2);

        var memoriesCountImage = BitmapHelper.GetSubBitmap(
            Visualisation.GetBitmapFromMiniColumsMemoriesCount(model.Cortex),
            model.Cortex.MiniColumns.Dimensions[0] / 2,
            model.Cortex.MiniColumns.Dimensions[1] / 2,
            model.Cortex.SubArea_MiniColumns_Radius + 2);

        var gradientBitmap0 = Visualisation.GetGradientBigBitmap(GeneratedGradientMatrix);
        var r = Visualisation.GetGeneratedLine_GradientMatrix(width, height, 0, angle / (2.0 * Math.PI));

        ImageWithDesc[] imageWithDescs = [
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(gradientBitmap0),
                    Desc = @"Полная картина градиентов" },
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(r.Item2),
                    Desc = @"Иллюстрация направления" },
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(activityColorImage),
                    Desc = @"Активность миниколонок (белый - максимум)" },
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(superActivityColorImage),
                    Desc = $"Суперактивность миниколонок (белый - максимум, синий - максимум со штрафом). Значение: {activitiyMaxInfo.MaxSuperActivity}" },                
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(memoriesColorImage),
                    Desc = @"Средний цвет накопленных воспоминаний в миниколонках" },
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(memoriesCountImage),
                    Desc = @"Количество воспоминаний в миниколонках" }
                ];

        ImagesSet1.MainItemsControl.ItemsSource = imageWithDescs;
    }
}