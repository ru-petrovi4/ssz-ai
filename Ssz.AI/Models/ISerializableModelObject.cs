using Ssz.Utils.Serialization;

namespace Ssz.AI.Models
{
    /// <summary>
    ///     Typical life cycle:
    ///     -  Constructor
    ///     -  GenereateOwnedData(...) or DeserializeOwnedData(...)
    ///     -  Prepare(...)
    ///     -  Calculate(...), Calculates variables with State_ prefix
    ///     -  SerializeOwnedData(...)
    /// </summary>
    public interface ISerializableModelObject : IOwnedDataSerializable
    {        
    }
}
