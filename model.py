def get_response(prompt):
    from google import genai 
    client = genai.Client(api_key="AIzaSyC0eikGjv1HEX7Lx6-vPygkMcbYUcQ7cpE") 
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt,
    ) 
    return response.text
