package com.example.powerballapp

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.*
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.Body
import retrofit2.http.POST

// ====== Data class para request y response ======
data class PredictRequest(val input_numbers: List<Int>)
data class PredictResponse(val input: List<Int>, val suggested_numbers: List<Double>)

// ====== Interfaz de Retrofit ======
interface PowerballApi {
    @POST("predict/")
    suspend fun predict(@Body request: PredictRequest): PredictResponse
}

class MainActivity : AppCompatActivity() {

    private val api by lazy {
        Retrofit.Builder()
            .baseUrl("https://tu_dominio.onrender.com/") // Cambia a tu dominio
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(PowerballApi::class.java)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val inputEditText = findViewById<EditText>(R.id.inputNumbers)
        val predictBtn = findViewById<Button>(R.id.btnPredict)
        val resultView = findViewById<TextView>(R.id.resultView)

        predictBtn.setOnClickListener {
            val text = inputEditText.text.toString()
            val numbers = text.split(",").mapNotNull { it.trim().toIntOrNull() }

            if (numbers.size != 8) {
                resultView.text = "Debes ingresar 8 números!"
                return@setOnClickListener
            }

            // Coroutines para hacer request en background
            CoroutineScope(Dispatchers.IO).launch {
                try {
                    val response = api.predict(PredictRequest(numbers))
                    withContext(Dispatchers.Main) {
                        resultView.text = "Sugeridos: ${response.suggested_numbers}"
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        resultView.text = "Error: ${e.message}"
                    }
                }
            }
        }
    }
}
