using Avalonia.Media;
using Ssz.Utils;
using Ssz.Utils.Avalonia.Model3D;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Collections;
using TorchSharp;
using static TorchSharp.torch;
using Ssz.AI.Helpers;
using Microsoft.Extensions.Logging;
using System.Diagnostics;
using Serilog.Data;
using Ssz.Utils.Optimized;

namespace Ssz.AI.Models.MiniColumnDetailedModel;

public sealed class MiniColumnDetailed_CombinatorinalSpace : IDisposable
{
    #region construction and destruction
    
    public MiniColumnDetailed_CombinatorinalSpace(Random random, IRetinaConstants constants)
    {
        _random = random;
        _device = cuda.is_available() ? CUDA : CPU;

        PyramidalAxons = new Axon[PyramidalAxonsCount];

        // ----------------------------------------------------------
        //  ШАГ 1: Разместить 200 сом в цилиндре миниколонки.
        //  Сомы распределены по всем слоям (Z = 0..2000 мкм)
        //  и в радиусе ColumnRadiusUm от центральной оси.
        // ----------------------------------------------------------
        var pyramidalSomaPositions = GeneratePyramidalSomaPositions();

        // ----------------------------------------------------------
        //  ШАГ 2: Для каждого нейрона вырастить аксон.
        // ----------------------------------------------------------
        for (int i = 0; i < PyramidalAxonsCount; i += 1)
        {
            PyramidalAxons[i] = GrowPyramidalAxon(pyramidalSomaPositions[i], i);
        }

        ThalamocorticalAxons = ThalamocorticalAxonGenerator.Generate(random, constants.HashLength);

        Temp_ThalamocorticalZones_Cache = new(new FloatArrayComparer(constants.HashLength + 2));
    }

    public void Dispose()
    {
        _gridTensorBuffer?.Dispose();
    }

    #endregion

    #region public functions

    // ----------------------------------------------------------
    //  ФИЗИЧЕСКИЕ РАЗМЕРЫ МИНИКОЛОНКИ (мкм)
    // ----------------------------------------------------------

    /// <summary>Радиус миниколонки в горизонтальной плоскости (мкм).</summary>
    public const float MiniColumnRadiusUm = 20.0f;   // диаметр ~40 мкм

    /// <summary>
    ///    Радиус, внутри которого учитываются синапсы пирамидальных нейронов.
    /// </summary>
    public const float MiniColumnRadius_Extended_Um = 25.0f;   // диаметр ~40 мкм

    /// <summary>Высота миниколонки (мкм), покрывает слои II–VI.</summary>
    public const float MiniColumnHeightUm = 2000.0f;

    // ----------------------------------------------------------
    //  ПАРАМЕТРЫ АКСОНОВ
    // ----------------------------------------------------------

    /// <summary>Число аксонов в миниколонке.</summary>
    public const int PyramidalAxonsCount = 0;

    public readonly Axon[] PyramidalAxons;

    public readonly Axon[] ThalamocorticalAxons;

    /// <summary>Активные зоны пирамидальных аксонов (Зоны 1).</summary>
    public FastList<ActiveZone>? Temp_PyramidalZones;

    /// <summary>Активные зоны входящих таламокортикальных аксонов (Зоны 2).</summary>
    public FastList<ActiveZone>? Temp_ThalamocorticalZones;

    public readonly Dictionary<float[], FastList<ActiveZone>?> Temp_ThalamocorticalZones_Cache;

    /// <summary>Зоны совместной активации: рядом есть и Зоны 1, и Зоны 2 (Зоны 3).</summary>
    public FastList<ActiveZone>? Temp_ConvergenceZones;

    /// <summary>
    /// Вычисляет зоны активных синапсов раздельно для пирамидальных аксонов
    /// и таламокортикальных афферентов, а затем находит их зоны конвергенции.
    /// </summary>
    /// <param name="tcActivityBits">Активность локальных аксонов (длина AxonCount).</param>
    /// <param name="radiusUm">Радиус поиска для отдельных зон (мкм).</param>
    /// <param name="minActiveAxons">Минимум уникальных активных аксонов в одной зоне.</param>
    /// <param name="convergenceRadiusUm">
    ///   Максимальное расстояние между Зоной 1 и Зоной 2, при котором
    ///   они считаются зоной конвергенции (Зона 3). По умолчанию равно radiusUm.
    /// </param>
    public void FindActiveZones(
        float[] tcActivityBits,
        float radiusUm,
        int minActiveAxons,
        ILogger logger,        
        float convergenceRadiusUm = -1f)
    {
        if (convergenceRadiusUm < 0f)
            convergenceRadiusUm = radiusUm;

        using var disposeScope = torch.NewDisposeScope();

        // ----------------------------------------------------------
        //  ЗОНЫ 1: Пирамидальные аксоны
        // ----------------------------------------------------------
        _activePyramidalAxons.Clear(); // Пока не заполняем
        if (_activePyramidalAxons.Count >= minActiveAxons)
            Temp_PyramidalZones = ComputeActiveZones(
                    GetSynapsesByAxons(_activePyramidalAxons),                    
                    radiusUm,
                    minActiveAxons,
                    logger);
        else
            Temp_PyramidalZones = null;

        // ----------------------------------------------------------
        //  ЗОНЫ 2: Таламокортикальные аксоны
        // ----------------------------------------------------------
        _activeTcAxons.Clear();
        for (int i = 0; i < tcActivityBits.Length; i += 1)
        {
            if (i < ThalamocorticalAxons.Length)
            {
                var axon = ThalamocorticalAxons[i];
                axon.Temp_IsActive = (tcActivityBits[i] > 0.5f);
                if (axon.Temp_IsActive)
                    _activeTcAxons.Add(axon);
            }
        }
        float[] temp_ThalamocorticalZones_Cache_Key = new float[tcActivityBits.Length + 2];
        Array.Copy(tcActivityBits, temp_ThalamocorticalZones_Cache_Key, tcActivityBits.Length);
        temp_ThalamocorticalZones_Cache_Key[tcActivityBits.Length] = radiusUm;
        temp_ThalamocorticalZones_Cache_Key[tcActivityBits.Length + 1] = minActiveAxons;
        if (!Temp_ThalamocorticalZones_Cache.TryGetValue(temp_ThalamocorticalZones_Cache_Key, out Temp_ThalamocorticalZones))
        {            
            if (_activeTcAxons.Count >= minActiveAxons)
                Temp_ThalamocorticalZones = ComputeActiveZones(
                        GetSynapsesByAxons(_activeTcAxons),
                        radiusUm,
                        minActiveAxons,
                        logger);
            else
                Temp_ThalamocorticalZones = null;
            Temp_ThalamocorticalZones_Cache.Add(temp_ThalamocorticalZones_Cache_Key, Temp_ThalamocorticalZones);
        }

        // ----------------------------------------------------------
        //  ЗОНЫ 3: Конвергенция (Зоны 1 рядом с Зонами 2)
        // ----------------------------------------------------------
        Temp_ConvergenceZones = FindConvergenceZones(
            Temp_PyramidalZones,
            Temp_ThalamocorticalZones,
            convergenceRadiusUm);
    }

    #endregion    

    // ============================================================
    //  ГЕНЕРАЦИЯ ПОЗИЦИЙ СОМ
    // ============================================================
    /// <summary>
    /// Располагает 200 нейронных сом равномерно в цилиндре
    /// миниколонки. Диаметр ~40 мкм, высота ~2000 мкм.
    /// Сомы размещены в 5 слоях (II, III, IV, V, VI),
    /// что соответствует реальной корковой анатомии.
    /// </summary>
    private Vector3[] GeneratePyramidalSomaPositions()
    {
        // Границы слоёв (Z в мкм): слой II-III верхний, VI нижний
        // Примерные относительные толщины слоёв коры:
        // L II: 0–200, L III: 200–600, L IV: 600–900,
        // L V: 900–1400, L VI: 1400–2000
        ReadOnlySpan<(float zMin, float zMax, int neuronCount)> layers =
        [
            (-80f,      0f,    0),   // слой I
            (-200f,   -80f,   20),   // слой II
            (-600f,  -200f,   50),   // слой III
            (-900f,  -600f,   30),   // слой IV
            (-1400f,  -900f,  60),   // слой V
            (-2000f,  -1400f, 40),   // слой VI
        ];
        // Итого: 200 нейронов распределены по 5 слоям.

        var positions = new Vector3[PyramidalAxonsCount];
        int idx = 0;

        foreach (var (zMin, zMax, count) in layers)
        {
            for (int n = 0; n < count; n += 1)
            {
                // Равномерное распределение в круге (метод отбора)
                float x, y;
                do
                {
                    x = (_random.NextSingle() * 2.0f - 1.0f) * MiniColumnRadiusUm;
                    y = (_random.NextSingle() * 2.0f - 1.0f) * MiniColumnRadiusUm;
                }
                while (x * x + y * y > MiniColumnRadiusUm * MiniColumnRadiusUm);

                float z = zMin + _random.NextSingle() * (zMax - zMin);
                if (idx < positions.Length)
                    positions[idx] = new Vector3(x, y, z);
                idx += 1;
            }
        }

        return positions;
    }

    // ============================================================
    //  РОСТ АКСОНА
    // ============================================================    
    private Axon GrowPyramidalAxon(Vector3 somaPos, int axonIdx)
    {
        // TODO
        return new Axon(new AxonPoint(new Vector3()), new Synapse[0]);        
    }
    
    // ============================================================
    //  ВСПОМОГАТЕЛЬНЫЙ: СБОР СИНАПСОВ ПО СПИСКУ АКТИВНЫХ АКСОНОВ
    // ============================================================
    private Synapse[][] GetSynapsesByAxons(FastList<Axon> axons)
    {
        int count = axons.Count;
        var groups = new Synapse[count][];
        for (int i = 0; i < count; i += 1)
        {
            groups[i] = axons[i].Synapses;
        }
        return groups;
    }

    // ============================================================
    // ЯДРО ВЫЧИСЛЕНИЯ ЗОН ЧЕРЕЗ 3D-СВЁРТКУ (GPU/CPU)
    // ============================================================
    // ИСПРАВЛЕНИЯ:
    //
    // 1. voxelSizeUm теперь вычисляется адаптивно: min(radiusUm/2, 5 мкм).
    //    Было: фиксированное 5 мкм.
    //    При малом radiusUm (например, 3 мкм) ядро из 1 воксела не обеспечивало
    //    сферический поиск. Теперь шаг воксела всегда <= radiusUm/2, что гарантирует
    //    ядро минимум 5×5×5 при любом радиусе.
    //
    // 2. Back-project: исправлены индексы для тензора [D, H, W] (без batch-измерения).
    //    Было: axonsPerVoxel имел форму [1, D, H, W] из-за keepdim=false при сумме
    //    из [1, C, D, H, W] → фактически форма [1, D, H, W], dims=4, индексы 0..3.
    //    Исправление: явное squeeze(0) перед nonzero, чтобы форма стала [D, H, W],
    //    dims=3, индексы 0(z), 1(y), 2(x).
    //
    // 3. mergeRadSq теперь равен (voxelSizeUm * 1.5)² вместо radiusUm².
    //    Было: соседние валидные воксели на расстоянии voxelSizeUm сливались только
    //    если radiusUm достаточно большой — при малом radiusUm дедупликация не работала
    //    или зоны вдоль аксона не сливались в цепочку.
    //    Новое правило: сливаем только соседние воксели (расстояние <= 1.5 воксела),
    //    сохраняя топологически разнесённые кластеры как отдельные зоны.
    private FastList<ActiveZone> ComputeActiveZones(
        Synapse[][] activeSynapsesGroups,        
        float radiusUm,
        int minActiveAxons,
        ILogger logger)
    {
        int activeCount = activeSynapsesGroups.Length;

        // адаптивный размер воксела — не более radiusUm/2,
        // чтобы ядро свёртки всегда содержало >= 2 вокселя на радиус.
        // Нижний порог 1 мкм предотвращает взрывной рост числа вокселей.
        float voxelSizeUm = 1.0f; //Math.Max(1.0f, Math.Min(radiusUm / 2.0f, 5.0f));

        // ---- Bounding box ----
        SceneBounds bounds = new();
        for (int a = 0; a < activeCount; a += 1)
        {
            var activeSynapses = activeSynapsesGroups[a];
            for (int s = 0; s < activeSynapses.Length; s += 1)
                bounds.Update(activeSynapses[s].Position);
        }
        bounds.XMin -= radiusUm; bounds.YMin -= radiusUm; bounds.ZMin -= radiusUm;
        bounds.XMax += radiusUm; bounds.YMax += radiusUm; bounds.ZMax += radiusUm;

        int width = (int)MathF.Ceiling((bounds.XMax - bounds.XMin) / voxelSizeUm);
        int height = (int)MathF.Ceiling((bounds.YMax - bounds.YMin) / voxelSizeUm);
        int depth = (int)MathF.Ceiling((bounds.ZMax - bounds.ZMin) / voxelSizeUm);

        // ---- Grid tensor [1, activeCount, D, H, W] ----
        long totalVoxels = (long)activeCount * depth * height * width;

        if (_gridTensorBuffer is null)
            _gridTensorBuffer = new TensorBuffer(_device, totalVoxels);
        else
            _gridTensorBuffer.EnsureCapacity(totalVoxels);

        using var gridFlat_Cpu = _gridTensorBuffer.Tensor_Cpu_Buffer!.slice(0, 0, totalVoxels, 1);
        using var gridFlat_Device = _gridTensorBuffer.Tensor_device_Buffer.slice(0, 0, totalVoxels, 1);
        gridFlat_Cpu.zero_();

        long spatialSize = (long)depth * height * width;
        long areaSize = (long)height * width;
        var gridData = gridFlat_Cpu.data<float>();

        for (int a = 0; a < activeCount; a += 1)
        {
            var activeSynapses = activeSynapsesGroups[a];
            long channelOffset = (long)a * spatialSize;
            for (int s = 0; s < activeSynapses.Length; s += 1)
            {
                var p = activeSynapses[s].Position;
                int ix = (int)((p.X - bounds.XMin) / voxelSizeUm);
                int iy = (int)((p.Y - bounds.YMin) / voxelSizeUm);
                int iz = (int)((p.Z - bounds.ZMin) / voxelSizeUm);

                if (ix >= 0 && ix < width && iy >= 0 && iy < height && iz >= 0 && iz < depth)
                    gridData[channelOffset + (iz * areaSize) + ((long)iy * width) + ix] = 1.0f;
            }
        }

        gridFlat_Device.copy_(gridFlat_Cpu);
        using var gridTensor = gridFlat_Device.view(1, activeCount, depth, height, width);

        // ---- Spherical kernel ----
        int kRad = (int)MathF.Ceiling(radiusUm / voxelSizeUm);
        int kSize = kRad * 2 + 1;
        long weightElements = (long)activeCount * kSize * kSize * kSize;

        if (_weightTensorBuffer is null)
            _weightTensorBuffer = new TensorBuffer(_device, weightElements);
        else
            _weightTensorBuffer.EnsureCapacity(weightElements);

        using var weightFlat_Cpu = _weightTensorBuffer.Tensor_Cpu_Buffer!.slice(0, 0, weightElements, 1);
        using var weightFlat_Device = _weightTensorBuffer.Tensor_device_Buffer.slice(0, 0, weightElements, 1);
        weightFlat_Cpu.zero_();
        var kernelData = weightFlat_Cpu.data<float>();

        long kSpatial = (long)kSize * kSize * kSize;
        long kArea = (long)kSize * kSize;

        for (int kz = -kRad; kz <= kRad; kz += 1)
            for (int ky = -kRad; ky <= kRad; ky += 1)
                for (int kx = -kRad; kx <= kRad; kx += 1)
                {
                    float dist = MathF.Sqrt(kx * kx + ky * ky + kz * kz) * voxelSizeUm;
                    if (dist <= radiusUm)
                    {
                        long offset = (long)(kz + kRad) * kArea + (ky + kRad) * kSize + (kx + kRad);
                        for (int c = 0; c < activeCount; c += 1)
                            kernelData[c * kSpatial + offset] = 1.0f;
                    }
                }

        weightFlat_Device.copy_(weightFlat_Cpu);
        using var weightTensor = weightFlat_Device.view(activeCount, 1, kSize, kSize, kSize);

        // ---- Convolution ----
        using var convResult = nn.functional.conv3d(
            gridTensor, weightTensor,
            padding: new long[] { kRad, kRad, kRad },
            groups: activeCount);

        // convResult: [1, activeCount, D, H, W]
        // presentMask[c, z, y, x] = 1 если аксон c имеет хоть один синапс в сфере радиуса R
        using var presentMask = convResult.gt(0.0f).to_type(ScalarType.Float32);

        // Суммируем по каналам → число активных аксонов в каждом вокселе
        // keepdim: false: [1, C, D, H, W] → [1, D, H, W]
        using var axonsPerVoxel = presentMask.sum(new long[] { 1 }, keepdim: false);

        // убираем batch-измерение перед nonzero,
        // чтобы получить форму [D, H, W] и индексы (z, y, x) без сдвига.
        using var axonsPerVoxel3D = axonsPerVoxel.squeeze(0); // [D, H, W]
        using var validMask = axonsPerVoxel3D.ge(minActiveAxons);
        
        torch.cuda.synchronize();
        var sw = Stopwatch.StartNew();
        using var nonZeroIdx = validMask.nonzero();      // [N, 3]: (z, y, x)
        torch.cuda.synchronize();
        sw.Stop();
        logger.LogInformation("nonzero computed in {ElapsedMilliseconds} ms, found {Count} valid voxels",
            sw.ElapsedMilliseconds, nonZeroIdx.shape[0]);

        long numValid = nonZeroIdx.shape[0];
        long dims = nonZeroIdx.shape[1]; // должно быть 3

        long[] flatIdx = null!;
        if (numValid > 0)
            flatIdx = nonZeroIdx.data<long>().ToArray();

        // ---- Back-project ----
        // индексы (z=0, y=1, x=2)
        var rawCenters = new List<Vector3>((int)numValid);
        for (long i = 0; i < numValid; i += 1)
        {
            long zIdx = flatIdx[i * dims + 0];
            long yIdx = flatIdx[i * dims + 1];
            long xIdx = flatIdx[i * dims + 2];
            rawCenters.Add(new Vector3(
                bounds.XMin + xIdx * voxelSizeUm + voxelSizeUm * 0.5f,
                bounds.YMin + yIdx * voxelSizeUm + voxelSizeUm * 0.5f,
                bounds.ZMin + zIdx * voxelSizeUm + voxelSizeUm * 0.5f));
        }
        var finalZones = new FastList<ActiveZone>(512);
        for (int i = 0; i < rawCenters.Count; i += 1)
        {
            finalZones.Add(new ActiveZone { Center = rawCenters[i] });
        }

        //// ---- Deduplicate / merge ----
        //// mergeRadius = 1.5 воксела (сливаем только смежные воксели,
        //// не уничтожая топологически разнесённые зоны вдоль аксона).
        //float mergeRadius = voxelSizeUm * 1.5f;
        //float mergeRadSq = mergeRadius * mergeRadius;


        //bool[] merged = new bool[rawCenters.Count];

        //for (int i = 0; i < rawCenters.Count; i += 1)
        //{
        //    if (merged[i]) continue;

        //    Vector3 sumPos = rawCenters[i];
        //    int cnt = 1;
        //    merged[i] = true;

        //    for (int j = i + 1; j < rawCenters.Count; j += 1)
        //    {
        //        if (!merged[j]
        //            && Vector3.DistanceSquared(rawCenters[i], rawCenters[j]) <= mergeRadSq)
        //        {
        //            sumPos += rawCenters[j];
        //            cnt += 1;
        //            merged[j] = true;
        //        }
        //    }

        //    finalZones.Add(new ActiveZone { Center = sumPos / cnt });
        //}

        return finalZones;
    }

    // ============================================================
    //  ПОИСК ЗОН КОНВЕРГЕНЦИИ (Зоны 3)
    //
    //  Для каждой Зоны 2 ищем ближайшую Зону 1 в радиусе
    //  convergenceRadiusUm. Если находим — создаём Зону 3
    //  в середине между ними. Дубликаты отбрасываются.
    // ============================================================
    private static FastList<ActiveZone>? FindConvergenceZones(
        FastList<ActiveZone>? pyramidalZones,
        FastList<ActiveZone>? tcZones,
        float convergenceRadiusUm)
    {
        if (pyramidalZones is null || pyramidalZones.Count == 0)
            return null;
        if (tcZones is null || tcZones.Count == 0)
            return null;

        float radSq = convergenceRadiusUm * convergenceRadiusUm;
        var result = new FastList<ActiveZone>(64);
        bool[] tcUsed = new bool[tcZones.Count];

        for (int p = 0; p < pyramidalZones.Count; p += 1)
        {
            Vector3 pCenter = pyramidalZones[p].Center;

            for (int t = 0; t < tcZones.Count; t += 1)
            {
                if (tcUsed[t]) continue;

                float distSq = Vector3.DistanceSquared(pCenter, tcZones[t].Center);
                if (distSq <= radSq)
                {
                    // Зона конвергенции — центр между двумя зонами
                    result.Add(new ActiveZone
                    {
                        Center = (pCenter + tcZones[t].Center) * 0.5f
                    });
                    tcUsed[t] = true; // каждая ТК-зона участвует не более одного раза
                }
            }
        }

        return result.Count > 0 ? result : null;
    }
    
    private readonly Random _random;

    private readonly Device _device;    

    // Кэшированный тензор для переиспользования памяти
    private TensorBuffer? _gridTensorBuffer;

    private TensorBuffer? _weightTensorBuffer;

    private readonly FastList<Axon> _activePyramidalAxons = new FastList<Axon>(capacity: PyramidalAxonsCount);
    private readonly FastList<Axon> _activeTcAxons = new FastList<Axon>(256);
}

public class TensorBuffer : IDisposable
{
    public TensorBuffer(Device device, long capacity)
    {
        Device = device;
        Tensor_Buffer_Capacity = capacity; //Math.Max(totalVoxels, Tensor_Buffer_Capacity == 0 ? totalVoxels : Tensor_Buffer_Capacity * 2);

        Tensor_device_Buffer = empty(new long[] { Tensor_Buffer_Capacity }, dtype: ScalarType.Float32, device: device).DetachFromDisposeScope();
        Tensor_Cpu_Buffer = empty(new long[] { Tensor_Buffer_Capacity }, dtype: ScalarType.Float32, device: CPU).DetachFromDisposeScope();
    }

    public void EnsureCapacity(long capacity)
    {
        if (capacity <= Tensor_Buffer_Capacity)
            return;

        Tensor_device_Buffer.Dispose();
        Tensor_Cpu_Buffer.Dispose();

        Tensor_Buffer_Capacity = capacity; //Math.Max(totalVoxels, Tensor_Buffer_Capacity == 0 ? totalVoxels : Tensor_Buffer_Capacity * 2);

        Tensor_device_Buffer = empty(new long[] { Tensor_Buffer_Capacity }, dtype: ScalarType.Float32, device: Device).DetachFromDisposeScope();
        Tensor_Cpu_Buffer = empty(new long[] { Tensor_Buffer_Capacity }, dtype: ScalarType.Float32, device: CPU).DetachFromDisposeScope();
    }

    public readonly Device Device;

    public Tensor Tensor_device_Buffer;
    public Tensor Tensor_Cpu_Buffer;
    // Текущая вместимость (в элементах), чтобы понимать, когда нужно перевыделять память
    public long Tensor_Buffer_Capacity = 0;

    public void Dispose()
    {
        Tensor_device_Buffer?.Dispose();
        Tensor_Cpu_Buffer?.Dispose();
    }
}