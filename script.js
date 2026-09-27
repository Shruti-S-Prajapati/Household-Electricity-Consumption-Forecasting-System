const API_URL = "http://127.0.0.1:8000/predict";

async function predictConsumption() {

    const data = {

        temperature_c:
            parseFloat(
                document.getElementById("temperature").value
            ),

        humidity_pct:
            parseFloat(
                document.getElementById("humidity").value
            ),

        day_of_week:
            document.getElementById("day").value,

        is_weekend:
            parseInt(
                document.getElementById("weekend").value
            ),

        is_holiday:
            parseInt(
                document.getElementById("holiday").value
            ),

        people_at_home:
            parseInt(
                document.getElementById("people").value
            ),

        hours_at_home:
            parseFloat(
                document.getElementById("homeHours").value
            ),

        ac_hours:
            parseFloat(
                document.getElementById("ac").value
            ),

        fan_hours:
            parseFloat(
                document.getElementById("fan").value
            ),

        tv_hours:
            parseFloat(
                document.getElementById("tv").value
            ),

        washing_machine_used:
            parseInt(
                document.getElementById("washing").value
            )

    };


    try {

        const response = await fetch(
            API_URL,
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(data)
            }
        );


        if (!response.ok) {

            throw new Error(
                "Prediction request failed"
            );

        }


        const result =
            await response.json();


        document.getElementById("result")
            .innerText =
            result.predicted_electricity_consumption_kwh;


    }

    catch (error) {

        console.error(error);

        document.getElementById("result")
            .innerText =
            "Error connecting to API";

    }

}