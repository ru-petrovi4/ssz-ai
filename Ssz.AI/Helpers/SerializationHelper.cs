using Ssz.Utils.Serialization;
using System.IO;

namespace Ssz.AI.Helpers
{
    public static class SerializationHelper
    {
        public static void SaveToFile(string fileName, IOwnedDataSerializable ownedDataSerializable, object? context)
        {
            fileName = @"Data\" + fileName;
            using (FileStream stream = File.Create(fileName))
            using (var writer = new SerializationWriter(stream, false))
            {
                writer.WriteOwnedDataSerializable(ownedDataSerializable, context);
            }
        }

        public static void LoadFromFileIfExists(string fileName, IOwnedDataSerializable ownedDataSerializable, object? context)
        {
            fileName = @"Data\" + fileName;
            if (File.Exists(fileName))
            {
                using (var stream = new FileStream(fileName, FileMode.Open))
                using (var reader = new SerializationReader(stream))
                {
                    reader.ReadOwnedDataSerializable(ownedDataSerializable, context);
                }
            }
        }
    }
}
