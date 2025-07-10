from Crypto.Cipher import AES
from Crypto.Util import Counter
import binascii

# Hexadecimal data from the challenge
key_hex = "000000005c5470020000000031f4727bf7d4923400000000e7bbb1c900000000"
ciphertext_hex = "d4922d0bae1521cffe67ab68fd132595c4942d350ff30c906325faca486555dc2ac805afddd31f16401ab7e1ebccdfc7e765f6a76746e822dd7002a2691f9392"
nonce_hex = "010000000000000000000000"
counter_int = 1  # Assuming counter is 1, as indicated in the sample data

# Convert hex to bytes
key = binascii.unhexlify(key_hex)
ciphertext = binascii.unhexlify(ciphertext_hex)
nonce = binascii.unhexlify(nonce_hex)

# Set up the AES decryption with CTR mode
def decrypt_aes_ctr(ciphertext, key, nonce, counter_int):
    # Create a counter using nonce and the counter integer
    ctr = Counter.new(128, initial_value=int(binascii.hexlify(nonce), 16) + counter_int)

    # Initialize AES cipher in CTR mode
    cipher = AES.new(key, AES.MODE_CTR, counter=ctr)

    # Decrypt the ciphertext
    decrypted_data = cipher.decrypt(ciphertext)
    
    # Return the decrypted data as a string
    return decrypted_data

# Decrypt the ciphertext
decrypted_data = decrypt_aes_ctr(ciphertext, key, nonce, counter_int)

# Attempt to decode the decrypted data (assuming it's UTF-8 text)
try:
    decrypted_string = decrypted_data.decode('utf-8')
    print("Decrypted Flag:", decrypted_string)
except UnicodeDecodeError:
    print("Decryption successful, but the output is not a valid UTF-8 string.")

