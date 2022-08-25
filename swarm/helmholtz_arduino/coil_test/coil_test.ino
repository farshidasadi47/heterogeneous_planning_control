#include "coils.h"

Coils coil;
ProcessInput commander(coil);

void setup() 
{
    Serial.begin(115200);
    delay(500);
    // Initialize coils.
    coil.initialize();
}

void loop() 
{
    commander.get_command_n_set_coil();
    delay(10);
}
