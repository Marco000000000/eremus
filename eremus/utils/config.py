face_channels = [67,253,252,248,244,241,73,254,249,245,242,219,225,226,230,234,238,218,227,231,235,239,243,
    246,250,255,82,92,103,112,121,134,146,156,166,175,188,200,209,217,228,232,236,240,237,233,229,216,208,
    199,187,174,165,145,133,120,111,102,91,256,251,247,31,93,104,113,122,135,147,157,167,176,189,201]

channels = [i for i in range(256) if i+1 not in face_channels]
