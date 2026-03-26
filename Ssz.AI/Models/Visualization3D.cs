using Avalonia.Controls;
using Avalonia.Media;
using Ssz.AI.Models.MiniColumnDetailedModel;
using Ssz.AI.Views;
using Ssz.Utils;
using Ssz.Utils.Avalonia.Model3D;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace Ssz.AI.Models;

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

    public static Model3DScene Get_MiniColumnDetailed_Model3DScene(Ssz.AI.Models.MiniColumnDetailedModel.MiniColumnDetailed miniColumnDetailed)
    {
        Model3DScene model3DScene = new();        
                
        SceneBounds sceneBounds = new();

        model3DScene.Points = new List<Point3DWithColor>(1024);
        model3DScene.Lines = new List<List<Point3DWithColor>>(1024);

        if (miniColumnDetailed.Temp_ActiveZones is not null)
        {
            int displayCount = Math.Min(miniColumnDetailed.Temp_ActiveZones.Count, 10000);
            for (int i = 0; i < displayCount; i += 1)
            {
                var z = miniColumnDetailed.Temp_ActiveZones[i];

                sceneBounds.Update(z.Center);

                System.Drawing.Color color = System.Drawing.Color.White;
                model3DScene.Points.Add(new Point3DWithColor
                {
                    Position = z.Center,
                    Color = new System.Numerics.Vector4((float)color.R / 255, (float)color.G / 255, (float)color.B / 255, 1.0f)
                });
            }
        }

        for (int i = 0; i < miniColumnDetailed.Axons.Length; i += 1) //
        {
            var axon = miniColumnDetailed.Axons[i];
            model3DScene.Lines.AddRange(GetLines(null, axon.Root, ref sceneBounds, axon.Temp_IsActive));

            for (int j = 0; j < axon.Synapses.Length; j += 1)
            {
                var s = axon.Synapses[j];

                sceneBounds.Update(s.Position);

                //System.Drawing.Color color = System.Drawing.Color.Red;
                //model3DScene.Points.Add(new Point3DWithColor
                //{
                //    Position = s.Position,
                //    Color = new System.Numerics.Vector4((float)color.R / 255, (float)color.G / 255, (float)color.B / 255, 1.0f)
                //});
            }
        }

        sceneBounds.Normalize(model3DScene);

        return model3DScene;
    }

    public static List<List<Point3DWithColor>> GetLines(AxonPoint? preStartAxonPoint, AxonPoint startAxonPoint, ref SceneBounds sceneBounds, bool isActive)
    {        
        List<List<Point3DWithColor>> lines = new(1024);
        List<Point3DWithColor> line = new();

        System.Drawing.Color color;
        if (isActive)
            color = System.Drawing.Color.FromArgb(0x44, 0x44, 0xFF);
        else
            color = System.Drawing.Color.FromArgb(0x00, 0x00, 0xFF);

        if (preStartAxonPoint is not null)
            line.Add(new Point3DWithColor
            {
                Position = preStartAxonPoint.Position,
                Color = new System.Numerics.Vector4((float)color.R / 255, (float)color.G / 255, (float)color.B / 255, 1.0f)
            });
        lines.Add(line);
        AxonPoint axonPoint = startAxonPoint;
        for (; ; )
        {
            sceneBounds.Update(axonPoint.Position);
            
            line.Add(new Point3DWithColor
            {
                Position = axonPoint.Position,
                Color = new System.Numerics.Vector4((float)color.R / 255, (float)color.G / 255, (float)color.B / 255, 1.0f)
            });
            if (axonPoint.Next.Count == 0)
                break;
            if (axonPoint.Next.Count == 1)
            {
                axonPoint = axonPoint.Next[0];
                continue;
            }

            for (int i = 0; i < axonPoint.Next.Count; i += 1)
            {
                lines.AddRange(GetLines(axonPoint, axonPoint.Next[i], ref sceneBounds, isActive));
            }
            break;
        }
        return lines;
    }

    public static Model3DScene Get_MiniColumnsMemories_Model3DScene(Ssz.AI.Models.AdvancedEmbeddingModel2.Cortex cortex)
    {
        List<Point3DWithColor> point3DWithColorList = new();

        int mcxMin = Int32.MaxValue;
        int mcxMax = Int32.MinValue;
        int mcyMin = Int32.MaxValue;
        int mcyMax = Int32.MinValue;            

        foreach (int mc_index in Enumerable.Range(0, cortex.MiniColumns.Data.Length))
        {
            var mc = cortex.MiniColumns.Data[mc_index];
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
        model3DScene.Points = point3DWithColorList;
        return model3DScene;
    }

    public static Model3DScene GetSubArea_MiniColumnsMemories_Model3DScene(Cortex_Simplified cortex)
    {
        List<Point3DWithColor> point3DWithColorList = new();

        int mcxMin = Int32.MaxValue;
        int mcxMax = Int32.MinValue;
        int mcyMin = Int32.MaxValue;
        int mcyMax = Int32.MinValue;

        foreach (int mc_index in Enumerable.Range(0, cortex.SubArea_MiniColumns.Length))
        {
            Cortex_Simplified.MiniColumn mc = cortex.SubArea_MiniColumns[mc_index];
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

                double normalizedMagnitude = magnitude / cortex.Constants.MaxGradientMagnitudeExclusive; // 1448 - максимальная теоретическая магнитуда Собеля для 8-битных изображений (255 * sqrt(2))                                                                                                                                     //brightness = 0.5 + (1 - brightness) * 0.5;
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
        model3DScene.Points = point3DWithColorList;
        return model3DScene;
    }

    public static Model3DScene GetSubArea_MiniColumnsMemories_Model3DScene(Cortex cortex)
    {
        List<Point3DWithColor> points = new();

        int mcxMin = Int32.MaxValue;
        int mcxMax = Int32.MinValue;
        int mcyMin = Int32.MaxValue;
        int mcyMax = Int32.MinValue;

        foreach (int mc_index in Enumerable.Range(0, cortex.SubAreaOrAll_MiniColumns.Length))
        {
            Cortex.MiniColumn mc = cortex.SubAreaOrAll_MiniColumns[mc_index];
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

                double normalizedMagnitude = magnitude / cortex.Constants.MaxGradientMagnitudeExclusive; // 1448 - максимальная теоретическая магнитуда Собеля для 8-битных изображений (255 * sqrt(2))                                                                                                                                     //brightness = 0.5 + (1 - brightness) * 0.5;
                double saturation = 0.3 + normalizedMagnitude;
                if (saturation > 1)
                    saturation = 1;

                // Преобразуем угол из диапазона [-pi, pi] в диапазон [0, 1] для цвета
                float normalizedAngle = ((float)angle + MathF.PI) / (2 * MathF.PI);
                // Получаем цвет на основе угла градиента (можно использовать HSV, здесь упрощенный пример через цветовой спектр)
                System.Drawing.Color color = Visualisation.ColorFromHSV(normalizedAngle, saturation, 1);

                points.Add(new Point3DWithColor
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
        foreach (Point3DWithColor point3DWithColor in points)
        {
            point3DWithColor.Position.X = (float)(mcxMax - point3DWithColor.Position.X) / (float)(mcxMax - mcxMin) - 0.5f;
            point3DWithColor.Position.Y = (float)(mcyMax - point3DWithColor.Position.Y) / (float)(mcyMax - mcyMin) - 0.5f;
        }

        Model3DScene model3DScene = new();
        model3DScene.Points = points;
        return model3DScene;
    }    
}

public struct SceneBounds
{
    public SceneBounds()
    {
    }

    public float XMin = Int32.MaxValue;
    public float XMax = Int32.MinValue;
    public float YMin = Int32.MaxValue;
    public float YMax = Int32.MinValue;
    public float ZMin = Int32.MaxValue;
    public float ZMax = Int32.MinValue;

    public void Update(Vector3 v)
    {
        if (v.X > XMax)
            XMax = v.X;
        if (v.X < XMin)
            XMin = v.X;
        if (v.Y > YMax)
            YMax = v.Y;
        if (v.Y < YMin)
            YMin = v.Y;
        if (v.Z > ZMax)
            ZMax = v.Z;
        if (v.Z < ZMin)
            ZMin = v.Z;
    }

    public void Normalize(Model3DScene model3DScene)
    {
        // 1. Находим общий максимальный размах сцены по всем осям
        float maxRange = MathF.Max(XMax - XMin, MathF.Max(YMax - YMin, ZMax - ZMin));
        //if (maxRange <= 0f) maxRange = 0.000001f; // Защита от деления на ноль

        // 2. Вычисляем центр сцены
        Vector3 center = new Vector3(
            (XMax + XMin) / 2f,
            (YMax + YMin) / 2f,
            (ZMax + ZMin) / 2f
        );

        //float deltaMax = MathF.Max(MathF.Max(XMax - XMin, YMax - YMin), ZMax - ZMin);
        if (model3DScene.Points is not null)
            foreach (Point3DWithColor point3DWithColor in model3DScene.Points)
            {
                Normalize(ref point3DWithColor.Position, center, maxRange);
            }

        if (model3DScene.Lines is not null)
            foreach (var line in model3DScene.Lines)
                foreach (Point3DWithColor point3DWithColor in line)
                {
                    Normalize(ref point3DWithColor.Position, center, maxRange);
                }
    }

    public void Normalize(ref Vector3 v, Vector3 center, float maxRange)
    {
        v = (v - center) / maxRange;
    }
}
