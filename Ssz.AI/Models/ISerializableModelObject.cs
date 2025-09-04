using Ssz.Utils.Serialization;

namespace Ssz.AI.Models
{
    /// <summary>
    ///     Typical life cycle:
    ///     -  obj.Constructor
    ///     -  obj.GenerateOwnedData(...) or DeserializeOwnedData(...), Generates/Loads Owned Data.
    ///     -  obj.Prepare(...), Prepares Temp Data.
    ///     -  Calculate*(...), Calculates Owned Data and Temp Data.
    ///     -  obj.SerializeOwnedData(...), Saves Owned Data.
    /// </summary>
    public interface ISerializableModelObject : IOwnedDataSerializable
    {        
    }
}
