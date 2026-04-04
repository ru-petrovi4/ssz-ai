using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Ssz.Utils;

namespace Ssz.AI.Models.MiniColumnDetailedModel;

// ============================================================
// ТАЛАМОКОРТИКАЛЬНЫЙ АКСОН (ЛКТ → V1)
//
// Моделирует входящий афферентный аксон от латерального
// коленчатого тела (ЛКТ / LGN) к миниколонке первичной
// зрительной коры (V1).
//
// Четыре типа по классификации LGN-каналов:
//
// Magnocellular (M) → слой 4Cα (Z ≈ −600..−750 мкм)
//   - Движение, контраст, крупные рецептивные поля
//   - Горизонтальный арбор ~550–640 мкм диаметр
//   - ~6 490 синапсов на аксон (Blasdel & Lund 1983;
//     Freund et al. 1989; García-Marín et al. 2019)
//
// Parvocellular (P) → слой 4Cβ (Z ≈ −750..−900 мкм)
//   - Цвет, мелкие детали, медленное движение
//   - Горизонтальный арбор ~300–400 мкм диаметр
//   - ~3 500 синапсов (García-Marín et al. 2019, оценка)
//
// KoniocellularSuperficial (K_sup, K1/K2 ЛКТ)
//       → L1 + L3A (Z ≈ 0..−80 мкм и −80..−350 мкм)
//   - Диффузный арбор; ~120 синапсов
//   - Источник: Casagrande et al. 2007
//
// KoniocellularBlob (K_blob, K3–K6 ЛКТ)
//       → CO-блобы L3Bα (Z ≈ −350..−500 мкм)
//   - Компактный фокусированный арбор; ~217 синапсов
//   - Источник: Casagrande et al. 2007
//
// Изменение критерия захвата (v3):
//   Ранее точка входа аксона в WM ограничивалась радиусом
//   arborRadius + columnRadius (арбор достигает центра колонки).
//   Теперь: arborRadius + columnRadius + dendriticReach
//   (арбор достигает дендритов нейронов колонки).
//   Дендриты пирамидных клеток L3 V1 простираются
//   горизонтально на ~97 мкм от сомы.
//   Источник: Amatrudo et al. 2012, J. Neurosci. 32:1480–1491.
//
// Координатная система:
//   X, Y — горизонтальные оси (мкм)
//   Z = 0 — поверхность коры, отрицательные Z — глубже
//
// Источники:
//   - Blasdel & Lund 1983, J. Neurosci. 3:1389–1413
//   - Freund et al. 1989, J. Comp. Neurol. 289:315–336
//   - Callaway 1998, Annu. Rev. Neurosci. 21:47–74
//   - García-Marín et al. 2019, Cereb. Cortex 29:134–151
//   - Hendry & Reid 2000, Annu. Rev. Neurosci. 23:127–153
//   - Yabuta & Callaway 1998, J. Neurosci. 18:9489–9499
//   - Casagrande et al. 2007, Cereb. Cortex 17:2334–2345
//   - Amatrudo et al. 2012, J. Neurosci. 32:1480–1491
// ============================================================

public enum ThalamocorticalType
{
    Magnocellular             = 0,
    Parvocellular             = 1,
    KoniocellularSuperficial  = 2,
    KoniocellularBlob         = 3,
}

public sealed class ThalamocorticalAxon : IAxon
{
    // ----------------------------------------------------------
    // ПАРАМЕТРЫ АРБОРОВ ПО ТИПАМ
    // ----------------------------------------------------------

    private const float MArborRadiusUm      = 300.0f;
    private const int   MSynapsesCount      = 6_490;

    private const float PArborRadiusUm      = 175.0f;
    private const int   PSynapsesCount      = 3_500;

    private const float KSupArborRadiusUm   = 137.5f;
    private const int   KSupSynapsesCount   = 120;

    private const float KBlobArborRadiusUm  = 112.5f;
    private const int   KBlobSynapsesCount  = 217;

    // ----------------------------------------------------------
    // ЦЕЛЕВЫЕ Z-ДИАПАЗОНЫ ПО ТИПАМ (мкм)
    // Z=0 — поверхность, Z < 0 — глубже.
    // ----------------------------------------------------------

    private const float KSupL1ZMax   =    0.0f;
    private const float KSupL1ZMin   =  -80.0f;
    private const float KSupL3AZMax  =  -80.0f;
    private const float KSupL3AZMin  = -350.0f;

    private const float KBlobZMax    = -350.0f;
    private const float KBlobZMin    = -500.0f;
    private const float KBlobL4AZMax = -500.0f;
    private const float KBlobL4AZMin = -600.0f;

    private const float MTargetZMax  = -600.0f;
    private const float MTargetZMin  = -750.0f;

    private const float PTargetZMax  = -750.0f;
    private const float PTargetZMin  = -900.0f;

    // ----------------------------------------------------------
    // ДАННЫЕ АКСОНА
    // ----------------------------------------------------------
    
    public readonly ThalamocorticalType Type;
    public readonly AxonPoint           Root;
    public readonly Synapse[]           Synapses;
    public          bool                Temp_IsActive;

    AxonPoint IAxon.Root     => Root;
    Synapse[] IAxon.Synapses => Synapses;
    bool      IAxon.IsActive => Temp_IsActive;

    public ThalamocorticalAxon(
        ThalamocorticalType type,
        AxonPoint root, 
        Synapse[] synapses)
    {        
        Type     = type;
        Root     = root;
        Synapses = synapses;
    }

    // ============================================================
    // ФАБРИЧНЫЙ МЕТОД
    // ============================================================
    ///
    /// Генерирует таламокортикальный аксон заданного типа.
    ///
    /// <param name="index">Индекс в массиве афферентов.</param>
    /// <param name="type">Тип LGN-канала.</param>
    /// <param name="random">Генератор случайных чисел.</param>
    /// <param name="columnRadiusUm">Радиус миниколонки (мкм).</param>
    /// <param name="columnHeightUm">Высота миниколонки (мкм).</param>
    /// <param name="dendriticReachUm">
    ///   Горизонтальный радиус базальных дендритов пирамидных клеток (мкм).
    ///   Расширяет зону захвата аксонов: точка входа аксона в WM может
    ///   находиться на расстоянии до (arborRadius + columnRadius + dendriticReachUm)
    ///   от центра колонки. Значение по умолчанию 97 мкм
    ///   (Amatrudo et al. 2012, J. Neurosci. 32:1480).
    /// </param>
    public static ThalamocorticalAxon Generate(        
        ThalamocorticalType  type,
        Random               random,
        float                columnRadiusUm,
        float                columnHeightUm,
        float                dendriticReachUm = 97.0f)
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
            case ThalamocorticalType.KoniocellularSuperficial:
                arborRadius   = KSupArborRadiusUm;
                synapsesCount = KSupSynapsesCount;
                targetZMin    = KSupL1ZMin;
                targetZMax    = KSupL1ZMax;
                break;
            default: // KoniocellularBlob
                arborRadius   = KBlobArborRadiusUm;
                synapsesCount = KBlobSynapsesCount;
                targetZMin    = KBlobZMin;
                targetZMax    = KBlobZMax;
                break;
        }

        // ----------------------------------------------------------
        // ШАГ 1: Точка входа аксона в белом веществе (WM)
        //
        // Критерий: аксон включается, если его арбор достигает
        // дендритов ЛЮБОГО нейрона данной миниколонки.
        //
        // Геометрия:
        //   - Нейроны расположены внутри cylinder радиуса columnRadius.
        //   - Базальные дендриты простираются до dendriticReachUm.
        //   - Значит, дендриты занимают круг радиуса
        //     (columnRadius + dendriticReachUm) от центра колонки.
        //   - Арбор аксона достигает этой зоны, если точка входа
        //     аксона находится на расстоянии ≤
        //     (arborRadius + columnRadius + dendriticReachUm) от центра.
        //
        // Источник: Amatrudo et al. 2012, J. Neurosci. 32:1480–1491
        //   "Horizontal extent of basal dendrites: 194 ± 15 мкм в V1"
        // ----------------------------------------------------------

        float entryMaxRadius = arborRadius + columnRadiusUm + dendriticReachUm;

        float entryX, entryY;
        while (true)
        {
            entryX = (random.NextSingle() * 2.0f - 1.0f) * entryMaxRadius;
            entryY = (random.NextSingle() * 2.0f - 1.0f) * entryMaxRadius;

            float rSq = entryX * entryX + entryY * entryY;
            if (rSq > entryMaxRadius * entryMaxRadius) continue;
            break;
        }

        float     entryZ   = -columnHeightUm - 50.0f;
        var       entryPos = new Vector3(entryX, entryY, entryZ);
        var       root     = new AxonPoint(entryPos);

        // ----------------------------------------------------------
        // ШАГ 2: Вертикальный ствол (подъём из WM к целевому слою)
        // ----------------------------------------------------------

        float targetZ     = targetZMin + random.NextSingle() * (targetZMax - targetZMin);
        float ascentDist  = targetZ - entryZ;
        int   ascentSteps = Math.Max(4, (int)(ascentDist / 100.0f));
        float stepZ       = ascentDist / ascentSteps;

        AxonPoint current = root;
        float     jitter  = 2.0f;

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

        AxonPoint arborRoot = current;

        // ----------------------------------------------------------
        // ШАГ 3: Горизонтальный арбор в целевом слое
        // ----------------------------------------------------------

        switch (type)
        {
            case ThalamocorticalType.KoniocellularSuperficial:
                GrowKoniocellularSuperficialArbor(arborRoot, arborRadius, random);
                break;
            case ThalamocorticalType.KoniocellularBlob:
                GrowKoniocellularBlobArbor(arborRoot, arborRadius, random);
                break;
            default:
                GrowStandardArbor(arborRoot, arborRadius, targetZMin, targetZMax, random);
                break;
        }

        // ----------------------------------------------------------
        // ШАГ 4: Синаптические бутоны en passant
        // ----------------------------------------------------------

        Synapse[] synapses = PlaceSynapses(root, synapsesCount, columnRadiusUm, random);
        return new ThalamocorticalAxon(type, root, synapses);
    }

    // ============================================================
    // СТАНДАРТНЫЙ ГОРИЗОНТАЛЬНЫЙ АРБОР (M и P)
    // ============================================================

    private static void GrowStandardArbor(
        AxonPoint arborRoot,
        float     arborRadius,
        float     targetZMin,
        float     targetZMax,
        Random    random)
    {
        int primaryBranches = 2 + random.Next(0, 3);

        for (int b = 0; b < primaryBranches; b += 1)
        {
            float branchAngle   = (float)(b * Math.PI * 2.0 / primaryBranches)
                                  + (random.NextSingle() - 0.5f) * 0.4f;
            float branchLen     = arborRadius * (0.60f + random.NextSingle() * 0.20f);
            int   branchSteps   = Math.Max(4, (int)(branchLen / 40.0f));
            float branchStepXY  = branchLen / branchSteps;
            float verticalDrift = MathF.Abs(targetZMax - targetZMin) * 0.15f;

            AxonPoint branchCurrent = arborRoot;

            for (int s = 0; s < branchSteps; s += 1)
            {
                float curAngle = branchAngle + (random.NextSingle() - 0.5f) * 0.18f;
                var pos = new Vector3(
                    branchCurrent.Position.X + MathF.Cos(curAngle) * branchStepXY,
                    branchCurrent.Position.Y + MathF.Sin(curAngle) * branchStepXY,
                    Math.Clamp(
                        branchCurrent.Position.Z + (random.NextSingle() - 0.5f) * verticalDrift,
                        targetZMin, targetZMax)
                );
                var next = new AxonPoint(pos);
                branchCurrent.Next.Add(next);
                branchCurrent = next;
            }

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
                            targetZMin, targetZMax)
                    );
                    var next = new AxonPoint(pos);
                    subCurrent.Next.Add(next);
                    subCurrent = next;
                }
            }
        }
    }

    // ============================================================
    // АРБОР K_sup (K1/K2): L1 + L3A
    // ============================================================

    private static void GrowKoniocellularSuperficialArbor(
        AxonPoint arborRoot,
        float     arborRadius,
        Random    random)
    {
        int l1Branches = 2 + random.Next(0, 2);
        for (int b = 0; b < l1Branches; b += 1)
        {
            float angle  = (float)(b * Math.PI * 2.0 / l1Branches)
                           + (random.NextSingle() - 0.5f) * 0.5f;
            float len    = arborRadius * (0.50f + random.NextSingle() * 0.30f);
            int   steps  = Math.Max(3, (int)(len / 35.0f));
            float stepXY = len / steps;
            AxonPoint cur = arborRoot;
            for (int s = 0; s < steps; s += 1)
            {
                float a   = angle + (random.NextSingle() - 0.5f) * 0.20f;
                var pos = new Vector3(
                    cur.Position.X + MathF.Cos(a) * stepXY,
                    cur.Position.Y + MathF.Sin(a) * stepXY,
                    Math.Clamp(
                        cur.Position.Z + (random.NextSingle() - 0.5f) * 8.0f,
                        KSupL1ZMin, KSupL1ZMax)
                );
                var next = new AxonPoint(pos);
                cur.Next.Add(next);
                cur = next;
            }
        }

        if (random.NextSingle() < 0.80f)
        {
            float l3aTargetZ  = KSupL3AZMax + random.NextSingle() * (KSupL3AZMin - KSupL3AZMax);
            float descentDist = l3aTargetZ - arborRoot.Position.Z;
            int   descSteps   = Math.Max(3, (int)(MathF.Abs(descentDist) / 50.0f));
            float descStepZ   = descentDist / descSteps;
            AxonPoint descCur = arborRoot;
            for (int s = 0; s < descSteps; s += 1)
            {
                var pos = new Vector3(
                    descCur.Position.X + (random.NextSingle() - 0.5f) * 4.0f,
                    descCur.Position.Y + (random.NextSingle() - 0.5f) * 4.0f,
                    descCur.Position.Z + descStepZ
                );
                var next = new AxonPoint(pos);
                descCur.Next.Add(next);
                descCur = next;
            }

            int l3aBranches = 1 + random.Next(0, 2);
            for (int b = 0; b < l3aBranches; b += 1)
            {
                float angle  = random.NextSingle() * MathF.PI * 2.0f;
                float len    = arborRadius * (0.30f + random.NextSingle() * 0.30f);
                int   steps  = Math.Max(2, (int)(len / 35.0f));
                float stepXY = len / steps;
                AxonPoint brCur = descCur;
                for (int s = 0; s < steps; s += 1)
                {
                    float a   = angle + (random.NextSingle() - 0.5f) * 0.25f;
                    var pos = new Vector3(
                        brCur.Position.X + MathF.Cos(a) * stepXY,
                        brCur.Position.Y + MathF.Sin(a) * stepXY,
                        Math.Clamp(
                            brCur.Position.Z + (random.NextSingle() - 0.5f) * 15.0f,
                            KSupL3AZMin, KSupL3AZMax)
                    );
                    var next = new AxonPoint(pos);
                    brCur.Next.Add(next);
                    brCur = next;
                }
            }
        }
    }

    // ============================================================
    // АРБОР K_blob (K3–K6): CO-блобы L3Bα + редкие коллатерали
    // ============================================================

    private static void GrowKoniocellularBlobArbor(
        AxonPoint arborRoot,
        float     arborRadius,
        Random    random)
    {
        int primaryBranches = 2 + random.Next(0, 2);
        for (int b = 0; b < primaryBranches; b += 1)
        {
            float angle  = (float)(b * Math.PI * 2.0 / primaryBranches)
                           + (random.NextSingle() - 0.5f) * 0.4f;
            float len    = arborRadius * (0.55f + random.NextSingle() * 0.25f);
            int   steps  = Math.Max(3, (int)(len / 30.0f));
            float stepXY = len / steps;
            float vDrift = MathF.Abs(KBlobZMax - KBlobZMin) * 0.10f;

            AxonPoint cur = arborRoot;
            for (int s = 0; s < steps; s += 1)
            {
                float a   = angle + (random.NextSingle() - 0.5f) * 0.15f;
                var pos = new Vector3(
                    cur.Position.X + MathF.Cos(a) * stepXY,
                    cur.Position.Y + MathF.Sin(a) * stepXY,
                    Math.Clamp(
                        cur.Position.Z + (random.NextSingle() - 0.5f) * vDrift,
                        KBlobZMin, KBlobZMax)
                );
                var next = new AxonPoint(pos);
                cur.Next.Add(next);
                cur = next;
            }

            int subBranches = 1 + random.Next(0, 2);
            for (int sb = 0; sb < subBranches; sb += 1)
            {
                float subAngle  = angle + (random.NextSingle() - 0.5f) * 1.0f;
                float subLen    = arborRadius * (0.15f + random.NextSingle() * 0.20f);
                int   subSteps  = Math.Max(2, (int)(subLen / 30.0f));
                float subStepXY = subLen / subSteps;
                AxonPoint subCur = arborRoot;
                for (int s = 0; s < subSteps; s += 1)
                {
                    float a   = subAngle + (random.NextSingle() - 0.5f) * 0.15f;
                    var pos = new Vector3(
                        subCur.Position.X + MathF.Cos(a) * subStepXY,
                        subCur.Position.Y + MathF.Sin(a) * subStepXY,
                        Math.Clamp(
                            subCur.Position.Z + (random.NextSingle() - 0.5f) * vDrift,
                            KBlobZMin, KBlobZMax)
                    );
                    var next = new AxonPoint(pos);
                    subCur.Next.Add(next);
                    subCur = next;
                }
            }
        }

        if (random.NextSingle() < 0.10f)
            GrowKBlobCollateral(arborRoot, random, KSupL1ZMin, KSupL1ZMax, arborRadius * 0.30f);

        if (random.NextSingle() < 0.10f)
            GrowKBlobCollateral(arborRoot, random, KBlobL4AZMin, KBlobL4AZMax, arborRadius * 0.25f);
    }

    // ============================================================
    // КОЛЛАТЕРАЛЬ K_blob В НЕЦЕЛЕВОМ СЛОЕ
    // ============================================================

    private static void GrowKBlobCollateral(
        AxonPoint parentNode,
        Random    random,
        float     toZMin,
        float     toZMax,
        float     collateralRadius)
    {
        float targetZ    = toZMin + random.NextSingle() * (toZMax - toZMin);
        float ascentDist = targetZ - parentNode.Position.Z;
        int   steps      = Math.Max(3, (int)(MathF.Abs(ascentDist) / 50.0f));
        float stepZ      = ascentDist / steps;

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

        float angle  = random.NextSingle() * MathF.PI * 2.0f;
        float len    = collateralRadius * (0.5f + random.NextSingle() * 0.5f);
        int   bSteps = Math.Max(2, (int)(len / 30.0f));
        float stepXY = len / bSteps;
        AxonPoint bCur = current;
        for (int s = 0; s < bSteps; s += 1)
        {
            float a   = angle + (random.NextSingle() - 0.5f) * 0.25f;
            var pos = new Vector3(
                bCur.Position.X + MathF.Cos(a) * stepXY,
                bCur.Position.Y + MathF.Sin(a) * stepXY,
                Math.Clamp(
                    bCur.Position.Z + (random.NextSingle() - 0.5f) * 8.0f,
                    toZMin, toZMax)
            );
            var next = new AxonPoint(pos);
            bCur.Next.Add(next);
            bCur = next;
        }
    }

    // ============================================================
    // РАЗМЕЩЕНИЕ СИНАПТИЧЕСКИХ БУТОНОВ EN PASSANT
    // ============================================================

    private static Synapse[] PlaceSynapses(
        AxonPoint root,
        int       totalSynapses,
        float     columnRadiusUm,
        Random    random)
    {
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

        if (totalLength <= 0f || segments.Count == 0)
            return synapses.ToArray();

        float step      = totalLength / totalSynapses;
        int   segIdx    = 0;
        float distInSeg = step * 0.5f;

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

            var synPos = new Vector3(
                basePos.X + (random.NextSingle() - 0.5f) * 2.0f,
                basePos.Y + (random.NextSingle() - 0.5f) * 2.0f,
                basePos.Z + (random.NextSingle() - 0.5f) * 1.0f
            );

            // TEMPCODE
            if (synPos.Z >= PTargetZMin && synPos.Z <= MTargetZMax)
                synapses.Add(new Synapse(synPos));
            distInSeg += step;
        }

        return synapses.ToArray();
    }
}
