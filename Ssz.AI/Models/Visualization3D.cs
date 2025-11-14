using Avalonia.Controls;
using Avalonia.Media;
using Ssz.AI.Views;
using Ssz.Utils.Avalonia.Model3D;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Ssz.AI.Models
{
    public static class Visualization3D
    {
        //public static void ShowPoints(System.Drawing.Image[] images)
        //{
        //    var window = new Window
        //    {
        //        Width = 1500,
        //        Height = 600,
        //        Content = new Model3DView()
        //    };

        //    window.Show();
        //}

        public static Model3DScene Get_MiniColumnsMemories_Model3DScene(Ssz.AI.Models.AdvancedEmbeddingModel2.Cortex cortex)
        {
            List<Point3DWithColor> point3DWithColorList = new();

            int mcxMin = Int32.MaxValue;
            int mcxMax = Int32.MinValue;
            int mcyMin = Int32.MaxValue;
            int mcyMax = Int32.MinValue;

            foreach (var mci in Enumerable.Range(0, cortex.MiniColumns.Data.Length))
            {
                var mc = cortex.MiniColumns.Data[mci];
                if (mc.MCX > mcxMax)
                    mcxMax = mc.MCX;
                if (mc.MCX < mcxMin)
                    mcxMin = mc.MCX;
                if (mc.MCY > mcyMax)
                    mcyMax = mc.MCY;
                if (mc.MCY < mcyMin)
                    mcyMin = mc.MCY;

                foreach (var mi in Enumerable.Range(0, mc.CortexMemories.Count))
                {
                    var cortexMemory = mc.CortexMemories[mi];
                    if (cortexMemory is null)
                        continue;
                    
                    System.Drawing.Color color = cortexMemory.DiscreteRandomVector_Color;

                    point3DWithColorList.Add(new Point3DWithColor
                    {
                        Position = new System.Numerics.Vector3(
                            mc.MCX,
                            mc.MCY,
                            color.GetHue() / 360.0f - 0.5f),
                        Color = new System.Numerics.Vector4((float)color.R / 255, (float)color.G / 255, (float)color.B / 255, 1.0f)
                    });
                }
            }

            // Normalize
            foreach (Point3DWithColor point3DWithColor in point3DWithColorList)
            {
                point3DWithColor.Position.X = (float)(mcxMax - point3DWithColor.Position.X) / (float)(mcxMax - mcxMin) - 0.5f;
                point3DWithColor.Position.Y = (float)(mcyMax - point3DWithColor.Position.Y) / (float)(mcyMax - mcyMin) - 0.5f;
            }

            Model3DScene model3DScene = new();
            model3DScene.Point3DWithColorArray = point3DWithColorList.ToArray();
            return model3DScene;
        }

        public static Model3DScene GetSubArea_MiniColumnsMemories_Model3DScene(Cortex_Simplified cortex)
        {
            List<Point3DWithColor> point3DWithColorList = new();

            int mcxMin = Int32.MaxValue;
            int mcxMax = Int32.MinValue;
            int mcyMin = Int32.MaxValue;
            int mcyMax = Int32.MinValue;

            foreach (var mci in Enumerable.Range(0, cortex.SubArea_MiniColumns.Length))
            {
                Cortex_Simplified.MiniColumn mc = cortex.SubArea_MiniColumns[mci];
                if (mc.MCX > mcxMax)
                    mcxMax = mc.MCX;
                if (mc.MCX < mcxMin)
                    mcxMin = mc.MCX;
                if (mc.MCY > mcyMax)
                    mcyMax = mc.MCY;
                if (mc.MCY < mcyMin)
                    mcyMin = mc.MCY;

                foreach (var mi in Enumerable.Range(0, mc.Memories.Count))
                {
                    Cortex_Simplified.Memory? memory = mc.Memories[mi];
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
                    float normalizedAngle = ((float)angle + MathF.PI) / (2 * MathF.PI);
                    // Получаем цвет на основе угла градиента (можно использовать HSV, здесь упрощенный пример через цветовой спектр)
                    System.Drawing.Color color = Visualisation.ColorFromHSV(normalizedAngle, saturation, 1);

                    point3DWithColorList.Add(new Point3DWithColor
                    {
                        Position = new System.Numerics.Vector3(
                            mc.MCX,
                            mc.MCY,
                            normalizedAngle - 0.5f),                        
                        Color = new System.Numerics.Vector4((float)color.R / 255, (float)color.G / 255, (float)color.B / 255, 1.0f)
                    });
                }
            }

            // Normalize
            foreach (Point3DWithColor point3DWithColor in point3DWithColorList)
            {
                point3DWithColor.Position.X = (float)(mcxMax - point3DWithColor.Position.X) / (float)(mcxMax - mcxMin) - 0.5f;
                point3DWithColor.Position.Y = (float)(mcyMax - point3DWithColor.Position.Y) / (float)(mcyMax - mcyMin) - 0.5f;
            }

            Model3DScene model3DScene = new();
            model3DScene.Point3DWithColorArray = point3DWithColorList.ToArray();
            return model3DScene;
        }

        public static Model3DScene GetSubArea_MiniColumnsMemories_Model3DScene(Cortex cortex)
        {
            List<Point3DWithColor> point3DWithColorList = new();

            int mcxMin = Int32.MaxValue;
            int mcxMax = Int32.MinValue;
            int mcyMin = Int32.MaxValue;
            int mcyMax = Int32.MinValue;

            foreach (var mci in Enumerable.Range(0, cortex.SubAreaOrAll_MiniColumns.Length))
            {
                Cortex.MiniColumn mc = cortex.SubAreaOrAll_MiniColumns[mci];
                if (mc.MCX > mcxMax)
                    mcxMax = mc.MCX;
                if (mc.MCX < mcxMin)
                    mcxMin = mc.MCX;
                if (mc.MCY > mcyMax)
                    mcyMax = mc.MCY;
                if (mc.MCY < mcyMin)
                    mcyMin = mc.MCY;

                foreach (var mi in Enumerable.Range(0, mc.Memories.Count))
                {
                    Cortex.Memory? memory = mc.Memories[mi];
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
                    float normalizedAngle = ((float)angle + MathF.PI) / (2 * MathF.PI);
                    // Получаем цвет на основе угла градиента (можно использовать HSV, здесь упрощенный пример через цветовой спектр)
                    System.Drawing.Color color = Visualisation.ColorFromHSV(normalizedAngle, saturation, 1);

                    point3DWithColorList.Add(new Point3DWithColor
                    {
                        Position = new System.Numerics.Vector3(
                            mc.MCX,
                            mc.MCY,
                            normalizedAngle - 0.5f),
                        Color = new System.Numerics.Vector4((float)color.R / 255, (float)color.G / 255, (float)color.B / 255, 1.0f)
                    });
                }
            }

            // Normalize
            foreach (Point3DWithColor point3DWithColor in point3DWithColorList)
            {
                point3DWithColor.Position.X = (float)(mcxMax - point3DWithColor.Position.X) / (float)(mcxMax - mcxMin) - 0.5f;
                point3DWithColor.Position.Y = (float)(mcyMax - point3DWithColor.Position.Y) / (float)(mcyMax - mcyMin) - 0.5f;
            }

            Model3DScene model3DScene = new();
            model3DScene.Point3DWithColorArray = point3DWithColorList.ToArray();
            return model3DScene;
        }
    }
}
