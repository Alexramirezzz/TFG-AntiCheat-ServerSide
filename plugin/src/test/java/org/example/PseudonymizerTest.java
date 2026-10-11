package org.example;

import org.junit.jupiter.api.Test;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;

class PseudonymizerTest {

    @Test
    void hmacMatchesRfc4231TestCase1() {
        // RFC 4231 test case 1: key = 20 x 0x0b, data = "Hi There"
        byte[] key = new byte[20];
        Arrays.fill(key, (byte) 0x0b);
        Pseudonymizer p = new Pseudonymizer(key);
        String mac = Pseudonymizer.hex(p.hmac("Hi There".getBytes(StandardCharsets.US_ASCII)));
        assertEquals("b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7", mac);
    }

    @Test
    void stableKeyedAndNotATruncatedUuid() {
        UUID u = UUID.fromString("123e4567-e89b-12d3-a456-426614174000");
        Pseudonymizer a = new Pseudonymizer(Pseudonymizer.decodeKey(Pseudonymizer.generateKey()));
        Pseudonymizer b = new Pseudonymizer(Pseudonymizer.decodeKey(Pseudonymizer.generateKey()));
        assertEquals(a.pseudonym(u), a.pseudonym(u));
        assertNotEquals(a.pseudonym(u), b.pseudonym(u));
        assertTrue(a.pseudonym(u).matches("p-[0-9a-f]{16}"));
        assertFalse(a.pseudonym(u).contains("123e4567"));
    }

    @Test
    void invalidKeysAreRejected() {
        assertNull(Pseudonymizer.decodeKey(""));
        assertNull(Pseudonymizer.decodeKey("not base64 !!"));
        assertNull(Pseudonymizer.decodeKey("c2hvcnQ="));            // "short": < 16 bytes
        assertThrows(IllegalArgumentException.class, () -> new Pseudonymizer(new byte[8]));
    }
}
