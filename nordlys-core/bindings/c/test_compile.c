#include "nordlys.h"

// Simple compilation test
int main() {
    Nordlys* nordlys = nordlys_create("test.json", NORDLYS_DEVICE_CPU);
    if (nordlys) {
        nordlys_destroy(nordlys);
    }
    return 0;
}
