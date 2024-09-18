using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json.Linq;
using Ssz.AI.Core.Grafana;
using Ssz.Utils;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Ssz.AI.Grafana
{
    [Route("grafana/json")]
    public partial class GrafanaJsonController : ControllerBase
    {
        #region construction and destruction

        public GrafanaJsonController(                        
            ILogger<GrafanaJsonController> logger,
            DataToDisplayHolder dataToDisplayHolder)
        {
            _logger = logger;
            _dataToDisplayHolder = dataToDisplayHolder;
        }

        #endregion

        #region public functions           

        public const string BooleanHisogram_Metric = @"BooleanHisogram";

        /// <summary>
        ///     Used for "Test connection" on the datasource config page.
        /// </summary>
        /// <returns></returns>
        [HttpGet]
        public Task<IActionResult> DatasourceHealthAsync()
        {
            return Task.FromResult<IActionResult>(StatusCode(200));
        }

        /// <summary>
        ///     Returns available metrics.
        /// </summary>
        /// <param name="listMetricsRequest"></param>
        /// <returns></returns>
        [HttpPost(@"metrics")]
        public IActionResult ListMetrics([FromBody] ListMetricsRequest listMetricsRequest)
        {
            var result = new List<ListMetricsResponse>
                    {
                        new ListMetricsResponse
                        {
                            Label = "Boolean Vector Info",
                            Value = BooleanHisogram_Metric,
                            Payloads = new List<ListMetricsResponsePayload>
                            {
                                //new ListMetricsResponsePayload
                                //{
                                //    Label = "N",
                                //    Name = N_PropertyName,
                                //    Type = ListMetricsResponsePayload.TypeEnum.Input,
                                //    Placeholder = "integer",
                                //    ReloadMetric = true,
                                //    Width = 40,
                                //},                                
                            }
                        },                        
                    };
            return Ok(result);
        }

        /// <summary>
        ///     Returns a list of metric payload options.
        /// </summary>
        /// <param name="listMetricPayloadOptionsRequest"></param>
        /// <returns></returns>
        [HttpPost(@"metric-payload-options")]
        public IActionResult ListMetricPayloadOptions([FromBody] ListMetricPayloadOptionsRequest listMetricPayloadOptionsRequest)
        {
            string exampleJson = """
        [{ 
          "label": "CPUUtilization",
          "value": "CPUUtilization"
        },{
          "label": "DiskReadIOPS",
          "value": "DiskReadIOPS"
        },{
          "label": "memory_freeutilization",
          "value": "memory_freeutilization"
        }]
        """;
            //var example = exampleJson != null
            //    ? JsonConvert.DeserializeObject<List<ListMetricPayloadOptionsResponse>>(exampleJson)
            //    : default(List<ListMetricPayloadOptionsResponse>);

            //var result = new List<ListMetricPayloadOptionsResponse>
            //{
            //    new ListMetricPayloadOptionsResponse
            //    {
            //        Value = "Value1",
            //        Payloads = new List<ListMetricsResponsePayloads>
            //        {
            //            new ListMetricsResponsePayloads
            //            {
            //                Name = "aaa"
            //            }
            //        }
            //    }
            //};
            //return new ObjectResult(result);

            return new ObjectResult(default(List<ListMetricPayloadOptionsResponse>));
        }

        /// <summary>
        ///     Returns panel data or annotations.
        /// </summary>
        /// <param name="queryRequest"></param>
        /// <returns></returns>
        [HttpPost(@"query")]
        public async Task<IActionResult> QueryAsync([FromBody] QueryRequest queryRequest)
        {
            if (queryRequest.Targets is null)
                return NoContent();

            List<QueryResponse> result = new();

            foreach (var target in queryRequest.Targets)
            {
                switch (target.Target)
                {
                    case BooleanHisogram_Metric:
                        result.Add(await Query_BooleanHisogram(queryRequest, target));
                        break;                    
                }
            }

            return Ok(result);
        }

        /// <summary>
        ///    Returns data for Variable of type Query.
        /// </summary>
        /// <param name="variableRequest"></param>
        /// <returns></returns>
        [HttpPost(@"variable")]
        public IActionResult Variable([FromBody] VariableRequest variableRequest)
        {
            string exampleJson = """
        [
          {"__text":"Label 1", "__value":"Value1"},
          {"__text":"Label 2", "__value":"Value2"},
          {"__text":"Label 3", "__value":"Value3"}
        ]
        """;
            //var example = exampleJson != null
            //    ? JsonConvert.DeserializeObject<VariableResponse>(exampleJson)
            //    : default(VariableResponse);            
            return Ok(exampleJson);
        }

        /// <summary>
        ///     Returning tag keys for ad hoc filters.
        /// </summary>
        /// <returns></returns>
        [HttpPost(@"tag-keys")]
        public IActionResult TagKeys()
        {
            string exampleJson = """
        [
            {"type":"string","text":"City"},
            {"type":"string","text":"Country"}
        ]
        """;
            //var example = exampleJson != null
            //    ? JsonConvert.DeserializeObject<List<TagKeysResponse>>(exampleJson)
            //    : default(List<TagKeysResponse>);            
            return Ok(exampleJson);
        }

        /// <summary>
        ///     Returning tag values for ad hoc filters.
        /// </summary>
        /// <param name="TagValuesRequest"></param>
        /// <returns></returns>
        /// <exception cref="NotImplementedException"></exception>
        [HttpPost(@"tag-values")]
        public IActionResult TagValues([FromBody] TagValuesRequest tagValuesRequest)
        {
            return new ObjectResult(new CaseInsensitiveDictionary<string>());
        }

        #endregion

        #region private functions

        private Task<QueryResponse> Query_BooleanHisogram(QueryRequest queryRequest, QueryRequestTarget queryRequestTarget)
        {
            var jsonElement = (JObject)queryRequestTarget.Payload!;
            //int n = new Any(jsonElement[N_PropertyName]?.ToString() ?? @"0").ValueAsInt32(false);            

            List<object[]> rows = new List<object[]>();
            //for (int i = 0; i <= n; i += 1)
            //{
            //    rows.Add([i.ToString(), ModelHelper.GetProbability(i, n, m)]);
            //}

            var queryResponse =
                new QueryResponse
                {
                    Target = BooleanHisogram_Metric,
                    //Datapoints = datapoints,
                    Type = QueryResponse.TypeEnum.Table,
                    Columns = new List<QueryResponseColumn>
                    {
                                new QueryResponseColumn { Text = @"OnesCount", Type = QueryResponseColumn.TypeEnum.String },
                                new QueryResponseColumn { Text = @"Probability", Type = QueryResponseColumn.TypeEnum.Number },
                    },
                    Rows = rows,
                };

            return Task.FromResult(queryResponse);
        }        

        #endregion

        #region private fields

        private readonly ILogger _logger;
        private readonly DataToDisplayHolder _dataToDisplayHolder;

        #endregion
    }    
}




//HttpContext.Request.EnableBuffering();
//HttpContext.Request.Body.Position = 0;
//var rawRequestBody = await new StreamReader(HttpContext.Request.Body).ReadToEndAsync();


// metrics
//string exampleJson = """
//[{
//  "label": "Describe metric list", // Optional. If the value is empty, use the value as the label
//  "value": "DescribeMetricList", // The value of the option.
//  "payloads": [{ // Configuration parameters of the payload.
//    "label": "Namespace", // The label of the payload. If the value is empty, use the value as the label.
//    "name": "namespace", // The name of the payload. If the value is empty, use the name as the label.
//    "type": "select", // If the value is select, the UI of the payload is a radio box. If the value is multi-select, the UI of the payload is a multi selection box; if the value is input, the UI of the payload is an input box; if the value is textarea, the UI of the payload is a multiline input box. The default is input.
//    "placeholder": "Please select namespace", // Input box / selection box prompt information.
//    "reloadMetric": true, // Whether to overload the metrics API after modifying the value of the payload.
//    "width": 10, // Set the input / selection box width to a multiple of 8px. 
//    "options": [{ // If the payload type is select / multi-select, the list is the configuration of the option list.
//      "label": "acs_mongodb", // The label of the payload select option.
//      "value": "acs_mongodb", // The label of the payload value.
//        },{
//          "label": "acs_rds",
//          "value": "acs_rds",
//        }]
//      },{
//        "name": "metric",
//        "type": "select"
//      },{
//        "name": "instanceId",
//        "type": "select"
//      }]
//},{
//  "value": "DescribeMetricLast",
//  "payloads": [{
//    "name": "namespace",
//    "type": "select"
//  },{
//    "name": "metric",
//    "type": "select"
//  },{
//    "name": "instanceId",
//    "type": "multi-select"
//  }]
//}]
//""";
////var example = exampleJson != null
////    ? JsonConvert.DeserializeObject<List<ListMetricsResponse>>(exampleJson)
////    : default(List<ListMetricsResponse>);   
///

//[DataContract]
//public class PlotlyData
//{
//    [DataMember(Name = "x", EmitDefaultValue = true)]
//    public float[] X { get; set; } = null!;

//    [DataMember(Name = "y", EmitDefaultValue = true)]
//    public float[] Y { get; set; } = null!;

//    /// <summary>
//    /// Gets or Sets Type
//    /// </summary>
//    [TypeConverter(typeof(CustomEnumConverter<ModeEnum>))]
//    [JsonConverter(typeof(Newtonsoft.Json.Converters.StringEnumConverter))]
//    public enum ModeEnum
//    {
//        /// <summary>
//        /// Enum TableEnum for table
//        /// </summary>
//        [EnumMember(Value = "markers")] // Attribute is not supported in System.Text.Json
//        Markers = 1
//    }

//    /// <summary>
//    /// Gets or Sets Mode
//    /// </summary>
//    [Required]
//    [DataMember(Name = "mode", EmitDefaultValue = true)]
//    public ModeEnum Mode { get; set; }

//    /// <summary>
//    /// Gets or Sets Type
//    /// </summary>
//    [TypeConverter(typeof(CustomEnumConverter<TypeEnum>))]
//    [JsonConverter(typeof(Newtonsoft.Json.Converters.StringEnumConverter))]
//    public enum TypeEnum
//    {
//        /// <summary>
//        /// Enum TableEnum for table
//        /// </summary>
//        [EnumMember(Value = "scatter")] // Attribute is not supported in System.Text.Json
//        Scatter = 1
//    }

//    /// <summary>
//    /// Gets or Sets Type
//    /// </summary>
//    [Required]
//    [DataMember(Name = "type", EmitDefaultValue = true)]
//    public TypeEnum Type { get; set; }
//}