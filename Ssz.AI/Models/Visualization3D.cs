using Avalonia.Controls;
using Avalonia.Media;
using Ssz.AI.Views;
using System;
using System.Collections.Generic;
using System.Linq;
using static Ssz.AI.Models.Cortex;

namespace Ssz.AI.Models
{
    public static class Visualization3D
    {
        //public static void ShowPoints(System.DrawingCore.Image[] images)
        //{
        //    var window = new Window
        //    {
        //        Width = 1500,
        //        Height = 600,
        //        Content = new Model3DView()
        //    };

        //    window.Show();
        //}

        public static Model3DScene GetSubArea_MiniColumnsMemories_Model3DScene(Cortex cortex)
        {
            List<Point3DWithColor> point3DWithColorList = new();

            foreach (var mci in Enumerable.Range(0, cortex.SubArea_MiniColumns.Length))
            {
                MiniColumn mc = cortex.SubArea_MiniColumns[mci];

                foreach (var mi in Enumerable.Range(0, mc.Memories.Count))
                {
                    Memory? memory = mc.Memories[mi];
                    if (memory is null)
                        continue;

                    double gradX = memory.PictureAverageGradientInPoint.GradX;
                    double gradY = memory.PictureAverageGradientInPoint.GradY;

                    double magnitude = Math.Sqrt(gradX * gradX + gradY * gradY);
                    double angle = Math.Atan2(gradY, gradX); // Угол в радианах    

                    double normalizedMagnitude = magnitude / cortex.Constants.GeneratedMaxGradientMagnitude; // 1448 - максимальная теоретическая магнитуда Собеля для 8-битных изображений (255 * sqrt(2))                                                                                                                                     //brightness = 0.5 + (1 - brightness) * 0.5;
                    double saturation = 0.3 + normalizedMagnitude;
                    if (saturation > 1)
                        saturation = 1;

                    // Преобразуем угол из диапазона [-pi, pi] в диапазон [0, 1] для цвета
                    double normalizedAngle = (angle + Math.PI) / (2 * Math.PI);
                    // Получаем цвет на основе угла градиента (можно использовать HSV, здесь упрощенный пример через цветовой спектр)
                    System.DrawingCore.Color color = Visualisation.ColorFromHSV(360 * normalizedAngle, saturation, 1);

                    point3DWithColorList.Add(new Point3DWithColor
                    {
                        Position = new System.Numerics.Vector3(
                            mc.MCX,
                            mc.MCY,
                            (float)normalizedAngle * 20),
                        Color = new System.Numerics.Vector4(color.A, color.R, color.G, color.B)
                    });
                }
            }

            Model3DScene model3DScene = new();
            model3DScene.Point3DWithColorArray = point3DWithColorList.ToArray();
            return model3DScene;
        }
    }
}
