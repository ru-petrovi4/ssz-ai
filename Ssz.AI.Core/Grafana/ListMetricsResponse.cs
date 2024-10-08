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
    public partial class ListMetricsResponse : IEquatable<ListMetricsResponse>
    {
        /// <summary>
        /// If the value is empty, use the value as the label
        /// </summary>
        /// <value>If the value is empty, use the value as the label</value>
        [DataMember(Name="label", EmitDefaultValue=false)]
        public string Label { get; set; } = @"";

        /// <summary>
        /// The value of the option.
        /// </summary>
        /// <value>The value of the option.</value>
        [Required]
        [DataMember(Name="value", EmitDefaultValue=false)]
        public string Value { get; set; } = @"";

        /// <summary>
        /// Configuration parameters of the payload.
        /// </summary>
        /// <value>Configuration parameters of the payload.</value>
        [Required]
        [DataMember(Name="payloads", EmitDefaultValue=false)]
        public List<ListMetricsResponsePayload>? Payloads { get; set; }

        /// <summary>
        /// Returns the string presentation of the object
        /// </summary>
        /// <returns>String presentation of the object</returns>
        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.Append("class ListMetricsResponse {\n");
            sb.Append("  Label: ").Append(Label).Append("\n");
            sb.Append("  Value: ").Append(Value).Append("\n");
            sb.Append("  Payloads: ").Append(Payloads).Append("\n");
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
            return obj.GetType() == GetType() && Equals((ListMetricsResponse)obj);
        }

        /// <summary>
        /// Returns true if ListMetricsResponse instances are equal
        /// </summary>
        /// <param name="other">Instance of ListMetricsResponse to be compared</param>
        /// <returns>Boolean</returns>
        public bool Equals(ListMetricsResponse? other)
        {
            if (other is null) return false;
            if (ReferenceEquals(this, other)) return true;

            return 
                (
                    Label == other.Label ||
                    Label != null &&
                    Label.Equals(other.Label)
                ) && 
                (
                    Value == other.Value ||
                    Value != null &&
                    Value.Equals(other.Value)
                ) && 
                (
                    Payloads == other.Payloads ||
                    Payloads != null &&
                    other.Payloads != null &&
                    Payloads.SequenceEqual(other.Payloads)
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
                    if (Label != null)
                    projectionCode = projectionCode * 59 + Label.GetHashCode();
                    if (Value != null)
                    projectionCode = projectionCode * 59 + Value.GetHashCode();
                    if (Payloads != null)
                    projectionCode = projectionCode * 59 + Payloads.GetHashCode();
                return projectionCode;
            }
        }

        #region Operators
        #pragma warning disable 1591

        public static bool operator ==(ListMetricsResponse? left, ListMetricsResponse? right)
        {
            return Equals(left, right);
        }

        public static bool operator !=(ListMetricsResponse? left, ListMetricsResponse? right)
        {
            return !Equals(left, right);
        }

        #pragma warning restore 1591
        #endregion Operators
    }
}
