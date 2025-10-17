



# Font discovery

Exact layout required being able to compute precise text extents. There are few system dapple uses try to locate the correct font, in order of robustness:

1. **fontconfig** (Linux/Unix) - The standard font configuration system on Linux
2. **fc-list command** (Linux/Unix) - Falls back to the fontconfig command-line tool
3. **matplotlib FontManager** (All platforms) - Cross-platform font discovery
4. **Directory scanning** (All platforms) - Scans platform-specific font directories


**fontconfig** and **manplotlib** are both optional dependencies, that are recommended for best results.
