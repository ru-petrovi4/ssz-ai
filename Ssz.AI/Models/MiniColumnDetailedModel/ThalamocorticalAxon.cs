using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
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
//  Magnocellular (M) → слой 4Cα (Z ≈ −600..−750 мкм)
//    - Движение, контраст, крупные рецептивные поля
//    - Горизонтальный арбор ~550–640 мкм диаметр
//    - ~6 490 синапсов на аксон (Blasdel & Lund 1983;
//      Freund et al. 1989; García-Marín et al. 2019)
//
//  Parvocellular (P) → слой 4Cβ (Z ≈ −750..−900 мкм)
//    - Цвет, мелкие детали, медленное движение
//    - Горизонтальный арбор ~300–400 мкм диаметр
//    - ~3 500 синапсов (García-Marín et al. 2019, оценка)
//
//  Koniocellular (K) → слой 1 (Z ≈ 0..−80 мкм) +
//                       blob-зоны L2/3 (Z ≈ −80..−250 мкм)
//    - Цвет (S-конус), диффузный входной сигнал
//    - Горизонтальный арбор ~200–300 мкм диаметр
//    - ~1 500 синапсов (Callaway 1998; Hendry & Reid 2000)
//
//  Морфология (Callaway 1998; Blasdel & Lund 1983):
//    1. Аксон приходит снизу из белого вещества.
//       Точка входа в WM случайна в пределах радиуса арбора
//       данного типа (не только под колонкой), т.к. один M-аксон
//       охватывает ~600 мкм и может принадлежать соседней колонке.
//    2. Поднимается вертикально до целевого слоя.
//    3. В целевом слое разворачивается горизонтально.
//    4. Формирует несколько (~2–4) плоских кустистых арборов.
//    5. Синаптические бутоны en passant по всей длине ветвей.
//
//  ИСПРАВЛЕНИЯ относительно предыдущей версии:
//    [Пункт 1] Точка входа (entryX,entryY) ограничена радиусом арбора
//              конкретного типа, а не columnRadiusUm (~20 мкм).
//              Это отражает реальность: аксон из WM подходит
//              к колонке сбоку, если его арбор большой.
//    [Пункт 2] blobTargetZ для K-аксонов исправлен на отрицательные
//              значения (−80..−300 мкм), что соответствует L2/3.
//              В предыдущей версии использовались +80..+300 мкм,
//              то есть выше поверхности коры — биологически невозможно.
//    [Пункт 5] Диапазон Z для K-аксонов расширен до 0..−250 мкм
//              (ранее только 0..−80 мкм / L1), чтобы охватить L3A.
//
//  Координатная система:
//    X, Y — горизонтальные оси (мкм)
//    Z = 0 — поверхность коры, отрицательные Z — глубже
//
//  Источники:
//    - Blasdel & Lund 1983, J. Neurosci. 3:1389–1413
//    - Freund et al. 1989, J. Comp. Neurol. 289:315–336
//    - Callaway 1998, Annu. Rev. Neurosci. 21:47–74
//    - García-Marín et al. 2019, Cereb. Cortex 29:134–151
//    - Hendry & Reid 2000, Annu. Rev. Neurosci. 23:127–153
//    - Yabuta & Callaway 1998, J. Neurosci. 18:9489–9499
// ============================================================

/// <summary>
/// Тип таламокортикального канала (LGN pathway).
/// Определяет целевой слой, морфологию арбора и число синапсов.
/// </summary>
public enum ThalamocorticalType
{
    /// <summary>Магноцеллюлярный путь: движение/контраст → L4Cα.</summary>
    Magnocellular = 0,
    /// <summary>Парвоцеллюлярный путь: цвет/детали → L4Cβ.</summary>
    Parvocellular = 1,
    /// <summary>Конийоцеллюлярный путь: S-конус цвет → L1 и blob L2/3.</summary>
    Koniocellular = 2,
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
    private const float MArborRadiusUm = 300.0f;
    private const int MSynapsesCount = 6_490;

    // Parvocellular: арбор ~350 мкм диаметр, ~3500 синапсов (оценка)
    // Источник: García-Marín et al. 2019
    private const float PArborRadiusUm = 175.0f;
    private const int PSynapsesCount = 3_500;

    // Koniocellular: арбор ~250 мкм диаметр, ~1500 синапсов (оценка)
    // Источник: Callaway 1998; Hendry & Reid 2000
    private const float KArborRadiusUm = 125.0f;
    private const int KSynapsesCount = 1_500;

    // ----------------------------------------------------------
    //  ЦЕЛЕВЫЕ Z-ДИАПАЗОНЫ ПО ТИПАМ (мкм)
    //  Система координат: Z=0 — поверхность коры, Z < 0 — глубже.
    //
    //  L1:   Z =    0 ..  −80 мкм
    //  L2/3: Z =  −80 .. −600 мкм  (blob-зоны: −80..−250 мкм)
    //  L4Cα: Z = −600 .. −750 мкм  ← M-вход
    //  L4Cβ: Z = −750 .. −900 мкм  ← P-вход
    //  L5:   Z = −900 ..−1400 мкм
    //  L6:   Z =−1400 ..−2000 мкм
    //
    //  [Пункт 5 ИСПРАВЛЕН]: K-аксоны теперь охватывают L1 + верх L2/3
    //  (0..−250 мкм), а не только L1 (0..−80 мкм).
    //  Источник: Hendry & Reid 2000; Yabuta & Callaway 1998
    // ----------------------------------------------------------

    // K: L1 + начало L2/3
    private const float KTargetZMax = 0.0f;    // поверхность
    private const float KTargetZMin = -250.0f; // нижняя граница L3A

    // M: слой 4Cα
    private const float MTargetZMax = -600.0f;
    private const float MTargetZMin = -750.0f;

    // P: слой 4Cβ
    private const float PTargetZMax = -750.0f;
    private const float PTargetZMin = -900.0f;

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

    AxonPoint IAxon.Root     => Root;
    Synapse[] IAxon.Synapses => Synapses;
    bool      IAxon.IsActive => Temp_IsActive;

    /// <summary>Создаёт аксон с уже построенным деревом и синапсами.</summary>
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
        int                  index,
        ThalamocorticalType  type,
        Random               random,
        float                columnRadiusUm,
        float                columnHeightUm,
        bool                 entryInNeighborRing = false)
    {
        // Выбираем параметры арбора в зависимости от типа канала
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
        //  ШАГ 1: Точка входа аксона в белом веществе (WM)
        //
        //  [ИСПРАВЛЕНИЕ — пункт 1]:
        //  Ранее точка входа ограничивалась columnRadiusUm (~20 мкм),
        //  т.е. аксон всегда начинался строго под своей колонкой.
        //  Это неверно: один M-аксон покрывает арбор диаметром ~600 мкм
        //  и может принадлежать нейрону ЛКТ, чей ствол проходит на
        //  расстоянии до arborRadius от центра данной колонки.
        //
        //  Теперь точка входа случайна в пределах радиуса арбора
        //  (entryMaxRadius = arborRadius), что биологически корректно:
        //  аксон поднимается из WM, немного смещённый от центра колонки,
        //  и всё равно достигает нужного слоя горизонтальным арбором.
        //  Источник: Blasdel & Lund 1983; García-Marín et al. 2019
        // ----------------------------------------------------------

        // Определяем диапазон разброса точки входа в зависимости от режима:
        //
        //   entryInNeighborRing = false (собственный аксон):
        //     Точка входа случайна внутри columnRadiusUm.
        //     Аксон «принадлежит» данной колонке.
        //
        //   entryInNeighborRing = true (соседский аксон):
        //     Точка входа случайна в КОЛЬЦЕ [columnRadiusUm .. arborRadius].
        //     Ствол аксона поднимается из-под соседней колонки, но его
        //     горизонтальный арбор (радиуса arborRadius) всё равно достигает
        //     данной колонки и оставляет синапсы внутри неё.
        //     Источник: Blasdel & Lund 1983 — M-арборы покрывают
        //     несколько миниколонок одновременно.
        float entryMinRadius = entryInNeighborRing ? columnRadiusUm : 0.0f;
        float entryMaxRadius = arborRadius + columnRadiusUm; // всегда ограничено радиусом арбора

        float entryX, entryY;
        while (true)
        {
            // Равномерно случайная точка в квадрате [-R, +R]²
            entryX = (random.NextSingle() * 2.0f - 1.0f) * entryMaxRadius;
            entryY = (random.NextSingle() * 2.0f - 1.0f) * entryMaxRadius;

            float rSq = entryX * entryX + entryY * entryY;

            // Принимаем точку только в допустимом кольце [min..max]
            if (rSq > entryMaxRadius * entryMaxRadius) continue;
            if (rSq < entryMinRadius * entryMinRadius) continue;
            break;
        }

        // Точка входа — глубже дна колонки на ~50 мкм (подход из WM)
        float entryZ = -columnHeightUm - 50.0f;
        var   entryPos = new Vector3(entryX, entryY, entryZ);
        var   root     = new AxonPoint(entryPos);

        // ----------------------------------------------------------
        //  ШАГ 2: Вертикальный ствол (ascent, подъём)
        //
        //  Аксон поднимается от WM (entryZ ≈ −2050 мкм) к целевому
        //  слою (targetZ ≈ −650..−900 мкм для M/P, 0..−250 мкм для K).
        //  Шаг 100 мкм, небольшие горизонтальные блуждания ±2 мкм.
        //  Источник: Callaway 1998 — ТК аксоны идут преимущественно
        //  вертикально через WM и нижние слои коры.
        // ----------------------------------------------------------

        // Случайный целевой Z внутри диапазона целевого слоя
        float targetZ = targetZMin + random.NextSingle() * (targetZMax - targetZMin);

        // Суммарное расстояние подъёма (положительное: Z растёт вверх к 0)
        float ascentDist  = targetZ - entryZ;
        int   ascentSteps = Math.Max(4, (int)(ascentDist / 100.0f));
        float stepZ       = ascentDist / ascentSteps;

        AxonPoint current = root;
        float     jitter  = 2.0f; // мкм горизонтального блуждания на шаг

        for (int s = 0; s < ascentSteps; s += 1)
        {
            var pos = new Vector3(
                current.Position.X + (random.NextSingle() - 0.5f) * jitter,
                current.Position.Y + (random.NextSingle() - 0.5f) * jitter,
                current.Position.Z + stepZ   // шаг вверх (к 0)
            );
            var next = new AxonPoint(pos);
            current.Next.Add(next);
            current = next;
        }

        // current теперь находится в целевом слое (Z ≈ targetZ)
        AxonPoint arborRoot = current;

        // ----------------------------------------------------------
        //  ШАГ 3: Горизонтальный арбор в целевом слое
        //
        //  2–4 первичных ветви расходятся равномерно по азимуту
        //  с небольшим случайным отклонением. Каждая ветвь
        //  имеет 1–3 вторичных подветви.
        //  Длина первичной ветви: 60–80% радиуса арбора.
        //  Источник: Blasdel & Lund 1983; García-Marín et al. 2019
        // ----------------------------------------------------------

        int primaryBranches = 2 + random.Next(0, 3); // 2–4 основных ветви

        for (int b = 0; b < primaryBranches; b += 1)
        {
            // Равномерное распределение азимутов + случайный джиттер ±0.2 рад
            float branchAngle = (float)(b * Math.PI * 2.0 / primaryBranches)
                                + (random.NextSingle() - 0.5f) * 0.4f;

            // Длина первичной ветви: 60–80% радиуса арбора
            float branchLen     = arborRadius * (0.60f + random.NextSingle() * 0.20f);
            int   branchSteps   = Math.Max(4, (int)(branchLen / 40.0f));
            float branchStepXY  = branchLen / branchSteps;

            // Вертикальный дрейф внутри слоя: 15% ширины слоя
            float verticalDrift = MathF.Abs(targetZMax - targetZMin) * 0.15f;

            AxonPoint branchCurrent = arborRoot;

            for (int s = 0; s < branchSteps; s += 1)
            {
                // Угол слегка меандрирует (±0.09 рад ~ ±5° на шаг)
                float curAngle = branchAngle + (random.NextSingle() - 0.5f) * 0.18f;

                var pos = new Vector3(
                    branchCurrent.Position.X + MathF.Cos(curAngle) * branchStepXY,
                    branchCurrent.Position.Y + MathF.Sin(curAngle) * branchStepXY,
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
                // Угол подветви: ответвляется от основного на ±0.6 рад
                float subAngle  = branchAngle + (random.NextSingle() - 0.5f) * 1.2f;
                float subLen    = arborRadius * (0.20f + random.NextSingle() * 0.25f);
                int   subSteps  = Math.Max(2, (int)(subLen / 40.0f));
                float subStepXY = subLen / subSteps;

                // Подветвь стартует из корня арбора (arborRoot)
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
        //  ШАГ 4 (только для K-аксонов): дополнительный арбор
        //  в blob-зонах L2/3
        //
        //  [ИСПРАВЛЕНИЕ — пункт 2]:
        //  В предыдущей версии blobTargetZ = +80..+300 мкм,
        //  что выше поверхности коры (физически невозможно).
        //  Исправлено: blobTargetZ = −80..−300 мкм (глубже
        //  поверхности, слои L2/3 blob-зоны).
        //  Источник: Hendry & Reid 2000; Yabuta & Callaway 1998
        // ----------------------------------------------------------

        if (type == ThalamocorticalType.Koniocellular)
        {
            // Blob-зона L2/3: Z = −80..−300 мкм (отрицательные значения!)
            float blobTargetZ = -(80.0f + random.NextSingle() * 220.0f);

            GrowKoniocellularBlobBranch(
                arborRoot,
                blobTargetZ,
                KArborRadiusUm * 0.6f,
                random,                
                -300.0f);     // toZ:   нижняя граница blob-зоны (−300 мкм)
        }

        // ----------------------------------------------------------
        //  ШАГ 5: Размещение синаптических бутонов равномерно
        //  по всей длине дерева аксона.
        // ----------------------------------------------------------

        Synapse[] synapses = PlaceSynapses(root, synapsesCount, columnRadiusUm, random);

        return new ThalamocorticalAxon(index, type, root, synapses);
    }

    // ============================================================
    //  ВСПОМОГАТЕЛЬНЫЙ МЕТОД: BLOB-ВЕТВЬ ДЛЯ K-АКСОНОВ
    // ============================================================
    /// <summary>
    /// Строит дополнительную ветвь K-аксона, уходящую вверх от L1
    /// к blob-зонам слоя 2/3 (Z = −80..−300 мкм).
    /// </summary>
    /// <param name="parentNode">Узел-родитель в арборе (в L1).</param>
    /// <param name="targetZ">Целевая глубина blob-зоны (мкм, отрицательная).</param>
    /// <param name="blobRadius">Радиус горизонтального арбора в blob-зоне (мкм).</param>
    /// <param name="random">Генератор случайных чисел.</param>
    /// <param name="toZ">Верхняя граница blob-арбора (мкм).</param>
    private static void GrowKoniocellularBlobBranch(
        AxonPoint parentNode,
        float     targetZ,
        float     blobRadius,
        Random    random,        
        float     toZ)
    {
        // Вертикальный «спуск» от L1 вглубь к blob-зоне L2/3.
        // fromZ ≈ −250 мкм (нижняя граница L1 после исправления),
        // targetZ ≈ −80..−300 мкм.
        // Разница (targetZ − fromZ) может быть положительной или
        // отрицательной в зависимости от случайного targetZ.
        float ascentDist = targetZ - parentNode.Position.Z;
        int   steps      = Math.Max(3, (int)(MathF.Abs(ascentDist) / 50.0f));
        float stepZ      = ascentDist / steps;

        AxonPoint current = parentNode;

        for (int s = 0; s < steps; s += 1)
        {
            // Небольшое горизонтальное блуждание ±1.5 мкм
            var pos = new Vector3(
                current.Position.X + (random.NextSingle() - 0.5f) * 3.0f,
                current.Position.Y + (random.NextSingle() - 0.5f) * 3.0f,
                current.Position.Z + stepZ
            );
            var next = new AxonPoint(pos);
            current.Next.Add(next);
            current = next;
        }

        // Горизонтальный арбор в blob-зоне: 2–3 ветви
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
                    Math.Clamp(
                        bCurrent.Position.Z + (random.NextSingle() - 0.5f) * 10.0f,
                        toZ,       // нижняя граница blob-зоны
                        targetZ + 60.0f) // верхняя граница
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
    /// по всей длине дерева аксона (от входа WM до концевых ветвей).
    /// Алгоритм: строит список отрезков, вычисляет суммарную длину,
    /// шагает с постоянным интервалом вдоль всего дерева.
    /// </summary>
    /// <param name="root">Корень дерева аксона.</param>
    /// <param name="totalSynapses">Количество синапсов для размещения.</param>
    /// <param name="columnRadiusUm">Радиус миниколонки (мкм).</param>
    /// <param name="random">Генератор случайных чисел.</param>
    private static Synapse[] PlaceSynapses(
        AxonPoint root,
        int       totalSynapses,
        float     columnRadiusUm,
        Random    random)
    {
        // --- Собираем все отрезки дерева (обход в глубину без рекурсии) ---
        var   segments    = new List<(Vector3 Start, Vector3 End, float Length)>(256);
        var   stack       = new Stack<AxonPoint>(64);
        float totalLength = 0f;

        stack.Push(root);

        while (stack.Count > 0)
        {
            var pt = stack.Pop();

            if (pt.Next != null)
            {
                for (int i = 0; i < pt.Next.Count; i += 1)
                {
                    var   child = pt.Next[i];
                    float len   = Vector3.Distance(pt.Position, child.Position);

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

        // Краевой случай: вырожденное дерево — всё в корне
        if (totalLength <= 0f || segments.Count == 0)
        {
            for (int s = 0; s < totalSynapses; s += 1)
                synapses.Add(new Synapse(root.Position));
            return synapses.ToArray();
        }

        // --- Равномерное размещение по длине ---
        // Шаг между синапсами: суммарная длина / число синапсов.
        // Первый синапс смещён на полшага от начала, чтобы
        // краевые области не были перегружены бутонами.
        float step       = totalLength / totalSynapses;
        int   segIdx     = 0;
        float distInSeg  = step * 0.5f; // начальное смещение

        for (int s = 0; s < totalSynapses; s += 1)
        {
            // Пропускаем отрезки, длина которых меньше текущего расстояния
            while (distInSeg > segments[segIdx].Length && segIdx < segments.Count - 1)
            {
                distInSeg -= segments[segIdx].Length;
                segIdx    += 1;
            }

            var   seg     = segments[segIdx];
            float t       = distInSeg / seg.Length;              // доля [0..1] по отрезку
            var   basePos = Vector3.Lerp(seg.Start, seg.End, t); // точная 3D-позиция

            // Случайное смещение ~1 мкм (биологический размер бутона en passant)
            var synPos = new Vector3(
                basePos.X + (random.NextSingle() - 0.5f) * 2.0f,
                basePos.Y + (random.NextSingle() - 0.5f) * 2.0f,
                basePos.Z + (random.NextSingle() - 0.5f) * 1.0f
            );

            // Фильтр: принимаем синапсы в пределах расширенного радиуса колонки
            if (MiniColumnDetailed.GetLengthXY(synPos) < MiniColumnDetailed.SynapsesRadiusUs)
                synapses.Add(new Synapse(synPos));

            distInSeg += step; // шагаем дальше
        }

        return synapses.ToArray();
    }
}
