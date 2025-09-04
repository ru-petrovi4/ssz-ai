using Ssz.Utils.Serialization;

namespace Ssz.AI.Models
{
    /// <summary>
    ///     Typical life cycle:
    ///     -  Constructor
    ///     -  GenerateOwnedData(...) or DeserializeOwnedData(...), Generates/Loads Owned Data.
    ///     -  Prepare(...), Prepares Owned Data and Temp Data.
    ///     -  Calculate*(...), Calculates Owned Data and Temp Data.
    ///     -  SerializeOwnedData(...), Saves Owned Data.
    /// </summary>
    public interface ISerializableModelObject : IOwnedDataSerializable
    {        
    }
}
