using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using Ssz.Utils;

namespace Ssz.AI.Models.MiniColumnDetailedModel;

// ============================================================
//  ТАЛАМОКОРТИКАЛЬНЫЙ АКСОН (ЛКТ → V1)
//
//  Моделирует входящий афферентный аксон от латерального
//  коленчатого тела (ЛКТ / LGN) к миниколонке первичной
//  зрительной коры (V1).
//
//  Три типа по классификации LGN-каналов:
//
//    Magnocellular (M) → слой 4Cα (Z ≈ 600–750 мкм)
//      - Движение, контраст, крупные рецептивные поля
//      - Горизонтальный арбор ~550–640 мкм диаметр
//      - ~6 490 синапсов на аксон (Blasdel & Lund 1983;
//        Freund et al. 1989; García-Marín et al. 2019)
//
//    Parvocellular (P) → слой 4Cβ (Z ≈ 750–900 мкм)
//      - Цвет, мелкие детали, медленное движение
//      - Горизонтальный арбор ~300–400 мкм диаметр
//      - ~3 500 синапсов (García-Marín et al. 2019, оценка)
//
//    Koniocellular (K) → слой 1 и blob-зоны слоя 2/3
//      - Цвет (S-конус), диффузный входной сигнал
//      - Горизонтальный арбор ~200–300 мкм диаметр
//      - ~1 500 синапсов
//
//  Морфология (Callaway 1998; Blasdel & Lund 1983):
//    1. Аксон приходит снизу (из белого вещества, Z > 2000 мкм)
//    2. Поднимается вертикально до целевого слоя
//    3. В целевом слое разворачивается горизонтально
//    4. Формирует несколько (~2–4) плоских кустистых арборов
//       (bouton clusters), каждый ~150–300 мкм в диаметре
//    5. Синаптические бутоны en passant по всей длине ветвей
//
//  Координатная система: совпадает с MiniColumnDetailed
//    X, Y — горизонтальные оси (мкм)
//    Z     — вертикальная ось (0 = поверхность, отрицательные Z глубже)
//
//  Источники:
//    - Blasdel & Lund 1983, J. Neurosci. 3:1389–1413
//    - Freund et al. 1989, J. Comp. Neurol. 289:315–336
//    - Callaway 1998, Annu. Rev. Neurosci. 21:47–74
//    - García-Marín et al. 2019, Cereb. Cortex 29:134–151
//    - Yabuta & Callaway 1998, J. Neurosci. 18:9489–9499
// ============================================================

/// <summary>
/// Тип таламокортикального канала (LGN pathway).
/// Определяет целевой слой, морфологию арбора и число синапсов.
/// </summary>
public enum ThalamocorticalType
{
    /// <summary>Магноцеллюлярный путь: движение/контраст → L4Cα.</summary>
    Magnocellular  = 0,
    /// <summary>Парвоцеллюлярный путь: цвет/детали → L4Cβ.</summary>
    Parvocellular  = 1,
    /// <summary>Конийоцеллюлярный путь: S-конус цвет → L1 и blob L2/3.</summary>
    Koniocellular  = 2,
}

/// <summary>
/// Таламокортикальный афферентный аксон от ЛКТ к миниколонке V1.
/// Моделирует пространственную траекторию и синаптические бутоны.
/// </summary>
public sealed class ThalamocorticalAxon : IAxon
{
    // ----------------------------------------------------------
    //  ПАРАМЕТРЫ АРБОРОВ ПО ТИПАМ (мкм и синапсы)
    // ----------------------------------------------------------

    // Magnocellular: арбор ~600 мкм диаметр, ~6490 синапсов
    // Источник: Blasdel & Lund 1983; García-Marín et al. 2019
    private const float MArborRadiusUm   = 300.0f;
    private const int   MSynapsesCount   = 6_490;

    // Parvocellular: арбор ~350 мкм диаметр, ~3500 синапсов (оценка)
    // Источник: García-Marín et al. 2019 (плотность P выше M в 4Cβ)
    private const float PArborRadiusUm   = 175.0f;
    private const int   PSynapsesCount   = 3_500;

    // Koniocellular: арбор ~250 мкм диаметр, ~1500 синапсов (оценка)
    // Источник: Callaway 1998; Hendry & Reid 2000
    private const float KArborRadiusUm   = 125.0f;
    private const int   KSynapsesCount   = 1_500;

    // ----------------------------------------------------------
    //  ЦЕЛЕВЫЕ Z-ДИАПАЗОНЫ ПО ТИПАМ (мкм, система V1)
    //  Слои в нашей СК (Z=0 поверхность, Z глубже):
    //    L1:    0–80
    //    L2/3:  80–600   (blob-зоны в верхней части)
    //    L4Cα:  600–750  (магноцеллюлярный вход)
    //    L4Cβ:  750–900  (парвоцеллюлярный вход)
    //    L5:    900–1400
    //    L6:    1400–2000
    // ----------------------------------------------------------
        
    private const float KTargetZMax = 0.0f;    // K также в blob L2/3 (Z~80–300)
    private const float KTargetZMin = -80.0f;     // L1
    private const float MTargetZMax = -600.0f;
    private const float MTargetZMin = -750.0f;   // L4Cα
    private const float PTargetZMax = -750.0f;
    private const float PTargetZMin = -900.0f;   // L4Cβ

    // ----------------------------------------------------------
    //  ДАННЫЕ АКСОНА
    // ----------------------------------------------------------

    /// <summary>Индекс аксона в массиве входящих афферентов.</summary>
    public readonly int Index;

    /// <summary>Тип канала LGN.</summary>
    public readonly ThalamocorticalType Type;

    /// <summary>Корневой узел дерева аксона (точка входа снизу из WM).</summary>
    public readonly AxonPoint Root;

    /// <summary>Все синаптические бутоны этого афферента.</summary>
    public readonly Synapse[] Synapses;

    /// <summary>Активность аксона: true — несёт входной сигнал.</summary>
    public bool Temp_IsActive;

    AxonPoint IAxon.Root => Root;

    Synapse[] IAxon.Synapses => Synapses;

    bool IAxon.IsActive => Temp_IsActive;

    public ThalamocorticalAxon(int index, ThalamocorticalType type,
                                AxonPoint root, Synapse[] synapses)
    {
        Index    = index;
        Type     = type;
        Root     = root;
        Synapses = synapses;
    }

    // ============================================================
    //  ФАБРИЧНЫЙ МЕТОД: генерация одного ТК-аксона
    // ============================================================
    /// <summary>
    /// Генерирует таламокортикальный афферентный аксон заданного типа.
    /// </summary>
    /// <param name="index">Индекс в массиве афферентов.</param>
    /// <param name="type">Тип LGN-канала.</param>
    /// <param name="random">Генератор случайных чисел.</param>
    /// <param name="columnRadiusUm">Радиус миниколонки (мкм).</param>
    /// <param name="columnHeightUm">Высота миниколонки (мкм).</param>
    /// <returns>Готовый ThalamocorticalAxon.</returns>
    public static ThalamocorticalAxon Generate(
        int                   index,
        ThalamocorticalType   type,
        Random                random,
        float                 columnRadiusUm,
        float                 columnHeightUm)
    {
        float arborRadius;
        int   synapsesCount;
        float targetZMin;
        float targetZMax;

        switch (type)
        {
            case ThalamocorticalType.Magnocellular:
                arborRadius   = MArborRadiusUm;
                synapsesCount = MSynapsesCount;
                targetZMin    = MTargetZMin;
                targetZMax    = MTargetZMax;
                break;
            case ThalamocorticalType.Parvocellular:
                arborRadius   = PArborRadiusUm;
                synapsesCount = PSynapsesCount;
                targetZMin    = PTargetZMin;
                targetZMax    = PTargetZMax;
                break;
            default: // Koniocellular
                arborRadius   = KArborRadiusUm;
                synapsesCount = KSynapsesCount;
                targetZMin    = KTargetZMin;
                targetZMax    = KTargetZMax;
                break;
        }

        // ----------------------------------------------------------
        //  ШАГ 1: Точка входа аксона в миниколонку
        //  Аксон приходит снизу из белого вещества.
        //  Точка входа — случайная позиция в пределах радиуса колонки
        //  на Z вниз columnHeightUm + небольшой запас.
        // ----------------------------------------------------------
        float entryX, entryY;
        do
        {
            entryX = (random.NextSingle() * 2.0f - 1.0f) * columnRadiusUm;
            entryY = (random.NextSingle() * 2.0f - 1.0f) * columnRadiusUm;
        }
        while (entryX * entryX + entryY * entryY > columnRadiusUm * columnRadiusUm);

        float entryZ = -columnHeightUm - 50.0f; // ~50 мкм ниже дна колонки
        var entryPos  = new Vector3(entryX, entryY, entryZ);
        var root      = new AxonPoint(entryPos);

        // ----------------------------------------------------------
        //  ШАГ 2: Вертикальный ствол (ascent)
        //  Аксон поднимается от WM к целевому слою.
        //  Небольшие горизонтальные отклонения (изгиб ~±3 мкм/100 мкм).
        //  Источник: Callaway 1998 — ТК аксоны идут преимущественно
        //  вертикально через белое вещество и нижние слои коры.
        // ----------------------------------------------------------
        float targetZ    = targetZMin + random.NextSingle() * (targetZMax - targetZMin);
        float ascentDist = targetZ - entryZ; // расстояние подъёма
        int   ascentSteps = Math.Max(4, (int)(ascentDist / 100.0f));
        float stepZ       = ascentDist / ascentSteps; // шаг вверх (увеличение Z)

        AxonPoint current = root;
        float     jitter  = 2.0f; // мкм горизонтального блуждания на шаг
        for (int s = 0; s < ascentSteps; s += 1)
        {
            var pos = new Vector3(
                current.Position.X + (random.NextSingle() - 0.5f) * jitter,
                current.Position.Y + (random.NextSingle() - 0.5f) * jitter,
                current.Position.Z + stepZ
            );
            var next = new AxonPoint(pos);
            current.Next.Add(next);
            current = next;
        }

        // current теперь находится в целевом слое (Z ≈ targetZ)
        AxonPoint arborRoot = current;

        // ----------------------------------------------------------
        //  ШАГ 3: Горизонтальный арбор в целевом слое
        //  Аксон ветвится на 2–4 основные ветви, которые
        //  расходятся горизонтально в плоскости целевого слоя.
        //  Каждая ветвь имеет дополнительные подветви (бутоны en passant).
        //  Источник: Blasdel & Lund 1983; García-Marín et al. 2019
        // ----------------------------------------------------------
        int primaryBranches = 2 + random.Next(0, 3); // 2–4 основных ветви

        for (int b = 0; b < primaryBranches; b += 1)
        {
            // Случайный азимутальный угол первичной ветви
            float branchAngle = (float)(b * Math.PI * 2.0 / primaryBranches)
                                + (random.NextSingle() - 0.5f) * 0.4f;

            // Длина первичной ветви: ~60–80% радиуса арбора
            float branchLen    = arborRadius * (0.60f + random.NextSingle() * 0.20f);
            int   branchSteps  = Math.Max(4, (int)(branchLen / 40.0f));
            float branchStepXY = branchLen / branchSteps;

            // Небольшой вертикальный дрейф в пределах слоя: ±10 мкм
            float verticalDrift = (targetZMax - targetZMin) * 0.15f;

            AxonPoint branchCurrent = arborRoot;
            for (int s = 0; s < branchSteps; s += 1)
            {
                float t     = (float)s / branchSteps;
                // Угол слегка меандрирует (±5° на шаг)
                float curAngle = branchAngle + (random.NextSingle() - 0.5f) * 0.18f;

                var pos = new Vector3(
                    branchCurrent.Position.X + MathF.Cos(curAngle) * branchStepXY,
                    branchCurrent.Position.Y + MathF.Sin(curAngle) * branchStepXY,
                    // Небольшой дрейф по Z внутри слоя
                    Math.Clamp(
                        branchCurrent.Position.Z + (random.NextSingle() - 0.5f) * verticalDrift,
                        targetZMin,
                        targetZMax)
                );
                var next = new AxonPoint(pos);
                branchCurrent.Next.Add(next);
                branchCurrent = next;
            }

            // Подветви (вторичные коллатерали): 1–3 на каждую первичную ветвь
            int subBranches = 1 + random.Next(0, 3);
            for (int sb = 0; sb < subBranches; sb += 1)
            {
                float subAngle  = branchAngle + (random.NextSingle() - 0.5f) * 1.2f;
                float subLen    = arborRadius * (0.20f + random.NextSingle() * 0.25f);
                int   subSteps  = Math.Max(2, (int)(subLen / 40.0f));
                float subStepXY = subLen / subSteps;

                AxonPoint subCurrent = arborRoot;
                for (int s = 0; s < subSteps; s += 1)
                {
                    float curAngle = subAngle + (random.NextSingle() - 0.5f) * 0.18f;
                    var pos = new Vector3(
                        subCurrent.Position.X + MathF.Cos(curAngle) * subStepXY,
                        subCurrent.Position.Y + MathF.Sin(curAngle) * subStepXY,
                        Math.Clamp(
                            subCurrent.Position.Z + (random.NextSingle() - 0.5f) * verticalDrift,
                            targetZMin,
                            targetZMax)
                    );
                    var next = new AxonPoint(pos);
                    subCurrent.Next.Add(next);
                    subCurrent = next;
                }
            }
        }

        // ----------------------------------------------------------
        //  KONIOCELLULAR: дополнительный арбор в blob-зонах L2/3
        //  K-аксоны проецируются как в L1, так и в blob-зоны (~Z 80–300 мкм).
        //  Источник: Hendry & Reid 2000, Annu. Rev. Neurosci. 23:127–153
        // ----------------------------------------------------------
        if (type == ThalamocorticalType.Koniocellular)
        {
            // Вторичная ветвь в blob-зону L2/3
            float blobTargetZ = 80.0f + random.NextSingle() * 220.0f; // ~80–300 мкм
            GrowKoniocellularBlobBranch(arborRoot, blobTargetZ, KArborRadiusUm * 0.6f,
                                        random, targetZMin, 300.0f);
        }

        // ----------------------------------------------------------
        //  ШАГ 4: Размещение синаптических бутонов
        // ----------------------------------------------------------
        Synapse[] synapses = PlaceSynapses(root, synapsesCount, columnRadiusUm, random);

        return new ThalamocorticalAxon(index, type, root, synapses);
    }

    // ============================================================
    //  ВСПОМОГАТЕЛЬНЫЙ МЕТОД: BLOB-ВЕТВЬ ДЛЯ K-АКСОНОВ
    // ============================================================
    private static void GrowKoniocellularBlobBranch(
        AxonPoint parentNode,
        float     targetZ,
        float     blobRadius,
        Random    random,
        float     fromZ,
        float     toZ)
    {
        // Короткий вертикальный подъём от L1 к blob-зоне L2/3
        float ascentDist  = fromZ - targetZ; // отрицательное (вверх)
        int   steps       = Math.Max(3, (int)(MathF.Abs(ascentDist) / 50.0f));
        float stepZ       = (targetZ - fromZ) / steps;

        AxonPoint current = parentNode;
        for (int s = 0; s < steps; s += 1)
        {
            var pos = new Vector3(
                current.Position.X + (random.NextSingle() - 0.5f) * 3.0f,
                current.Position.Y + (random.NextSingle() - 0.5f) * 3.0f,
                current.Position.Z + stepZ
            );
            var next = new AxonPoint(pos);
            current.Next.Add(next);
            current = next;
        }

        // Горизонтальный blob-арбор
        int primaryBranches = 2 + random.Next(0, 2);
        for (int b = 0; b < primaryBranches; b += 1)
        {
            float angle    = random.NextSingle() * MathF.PI * 2.0f;
            float len      = blobRadius * (0.5f + random.NextSingle() * 0.5f);
            int   subSteps = Math.Max(3, (int)(len / 35.0f));
            float stepXY   = len / subSteps;

            AxonPoint bCurrent = current;
            for (int s = 0; s < subSteps; s += 1)
            {
                float a = angle + (random.NextSingle() - 0.5f) * 0.25f;
                var pos = new Vector3(
                    bCurrent.Position.X + MathF.Cos(a) * stepXY,
                    bCurrent.Position.Y + MathF.Sin(a) * stepXY,
                    Math.Clamp(bCurrent.Position.Z + (random.NextSingle() - 0.5f) * 10.0f,
                                targetZ - 60.0f, targetZ + 60.0f)
                );
                var next = new AxonPoint(pos);
                bCurrent.Next.Add(next);
                bCurrent = next;
            }
        }
    }

    // ============================================================
    //  РАЗМЕЩЕНИЕ СИНАПТИЧЕСКИХ БУТОНОВ
    // ============================================================
    /// <summary>
    /// Равномерно расставляет синаптические бутоны en passant
    /// по всей длине арбора таламокортикального аксона.
    /// Бутоны размещаются только внутри (или вблизи) миниколонки.
    /// </summary>
    private static Synapse[] PlaceSynapses(
        AxonPoint root,
        int       totalSynapses,
        float     columnRadiusUm,
        Random    random)
    {
        // Собираем все отрезки дерева аксона
        var segments = new List<(Vector3 Start, Vector3 End, float Length)>(256);
        var stack    = new Stack<AxonPoint>(64);
        stack.Push(root);
        float totalLength = 0f;

        while (stack.Count > 0)
        {
            var pt = stack.Pop();
            if (pt.Next != null)
            {
                for (int i = 0; i < pt.Next.Count; i += 1)
                {
                    var child  = pt.Next[i];
                    float len  = Vector3.Distance(pt.Position, child.Position);
                    if (len > 0f)
                    {
                        segments.Add((pt.Position, child.Position, len));
                        totalLength += len;
                    }
                    stack.Push(child);
                }
            }
        }

        var synapses = new FastList<Synapse>(totalSynapses);

        if (totalLength <= 0f || segments.Count == 0)
        {
            for (int s = 0; s < totalSynapses; s += 1)
                synapses.Add(new Synapse(root.Position));
            return synapses.ToArray();
        }

        float step              = totalLength / totalSynapses;
        int   segIdx            = 0;
        float distInSeg         = step * 0.5f;        

        for (int s = 0; s < totalSynapses; s += 1)
        {
            while (distInSeg > segments[segIdx].Length && segIdx < segments.Count - 1)
            {
                distInSeg -= segments[segIdx].Length;
                segIdx    += 1;
            }

            var   seg     = segments[segIdx];
            float t       = distInSeg / seg.Length;
            var   basePos = Vector3.Lerp(seg.Start, seg.End, t);

            // Случайное смещение ~1 мкм (размер бутона en passant)
            var synPos = new Vector3(
                basePos.X + (random.NextSingle() - 0.5f) * 2.0f,
                basePos.Y + (random.NextSingle() - 0.5f) * 2.0f,
                basePos.Z + (random.NextSingle() - 0.5f) * 1.0f
            );

            // Фильтр: только бутоны внутри расширенного радиуса колонки
            if (MiniColumnDetailed.GetLengthXY(synPos) < MiniColumnDetailed.SynapsesRadiusUs)
                synapses.Add(new Synapse(synPos));

            distInSeg += step;
        }

        return synapses.ToArray();
    }
}
