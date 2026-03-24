using System.Numerics;

namespace Ssz.AI.Models.MiniColumnDetailedModel;

// ============================================================
//  СИНАПС
//  Хранит координату синаптического контакта (мкм).
//  System.Numerics.Vector3 — SIMD-совместимый тип.
// ============================================================
public sealed class Synapse
{
    /// <summary>3D-координата синапса в мкм.</summary>
    public Vector3 Position;

    public Synapse(Vector3 position)
    {
        Position = position;
    }
}
