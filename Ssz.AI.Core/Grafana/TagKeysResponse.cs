/*
 * simPod JSON Datasource API
 *
 * API definition for the Grafana plugin simpod json datasource https://github.com/simPod/grafana-json-datasource
 *
 * The version of the OpenAPI document: 0.1
 * 
 * Generated by: https://openapi-generator.tech
 */

using System;
using System.Linq;
using System.Text;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Runtime.Serialization;
using Newtonsoft.Json;
using Org.OpenAPITools.Converters;

namespace Ssz.AI.Core.Grafana
{ 
    /// <summary>
    /// 
    /// </summary>
    [DataContract]
    public partial class TagKeysResponse : IEquatable<TagKeysResponse>
    {
        /// <summary>
        /// Gets or Sets Type
        /// </summary>
        [DataMember(Name="type", EmitDefaultValue=false)]
        public string Type { get; set; } = @"";

        /// <summary>
        /// Gets or Sets Text
        /// </summary>
        [DataMember(Name="text", EmitDefaultValue=false)]
        public string Text { get; set; } = @"";

        /// <summary>
        /// Returns the string presentation of the object
        /// </summary>
        /// <returns>String presentation of the object</returns>
        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.Append("class TagKeysResponse {\n");
            sb.Append("  Type: ").Append(Type).Append("\n");
            sb.Append("  Text: ").Append(Text).Append("\n");
            sb.Append("}\n");
            return sb.ToString();
        }

        /// <summary>
        /// Returns true if objects are equal
        /// </summary>
        /// <param name="obj">Object to be compared</param>
        /// <returns>Boolean</returns>
        public override bool Equals(object? obj)
        {
            if (obj is null) return false;
            if (ReferenceEquals(this, obj)) return true;
            return obj.GetType() == GetType() && Equals((TagKeysResponse)obj);
        }

        /// <summary>
        /// Returns true if TagKeysResponse instances are equal
        /// </summary>
        /// <param name="other">Instance of TagKeysResponse to be compared</param>
        /// <returns>Boolean</returns>
        public bool Equals(TagKeysResponse? other)
        {
            if (other is null) return false;
            if (ReferenceEquals(this, other)) return true;

            return 
                (
                    Type == other.Type ||
                    Type != null &&
                    Type.Equals(other.Type)
                ) && 
                (
                    Text == other.Text ||
                    Text != null &&
                    Text.Equals(other.Text)
                );
        }

        /// <summary>
        /// Gets the projection code
        /// </summary>
        /// <returns>Projection code</returns>
        public override int GetHashCode()
        {
            unchecked // Overflow is fine, just wrap
            {
                var projectionCode = 41;
                // Suitable nullity checks etc, of course :)
                    if (Type != null)
                    projectionCode = projectionCode * 59 + Type.GetHashCode();
                    if (Text != null)
                    projectionCode = projectionCode * 59 + Text.GetHashCode();
                return projectionCode;
            }
        }

        #region Operators
        #pragma warning disable 1591

        public static bool operator ==(TagKeysResponse? left, TagKeysResponse? right)
        {
            return Equals(left, right);
        }

        public static bool operator !=(TagKeysResponse? left, TagKeysResponse? right)
        {
            return !Equals(left, right);
        }

        #pragma warning restore 1591
        #endregion Operators
    }
}
