# Scheme heavily adapted from https://github.com/deepmind/pycolab/
# Cleanup colors
# '@' means "wall"
# 'H' is potential waste spawn point
# 'R' is river cell
# 'S' is stream cell
# 'P' means "player" spawn point
# 'A' means apple spawn point
# 'B' is potential apple spawn point
# ' ' is empty space

# CLEANUP LIO
CLEANUP_N3_MAP = [
    '@@@@@@@@@@',
    '@HH   P B@',
    '@RR    BB@',
    '@HH     B@',
    '@RR    BB@',
    '@HH P   B@',
    '@RR    BB@',
    '@HH     B@',
    '@RRP   BB@',
    '@@@@@@@@@@']

# n=5
CLEANUP_N5_MAP = [
    '@@@@@@@@@@@@@@@@@@',
    '@RRRRRR     BBBBB@',
    '@HHHHHH    P BBBB@',
    '@RRRRRR     BBBBB@',
    '@RRRRR       BBBB@',
    '@RRRRR      BBBBB@',
    '@HHHH P      BBBB@',
    '@RRRRR      BBBBB@',
    '@HHHHHHSSSSSSBBBB@',
    '@HHHHHHSSSSSSBBBB@',
    '@RRRRR       BBBB@',
    '@HHHHH      BBBBB@',
    '@RRRRRR    P BBBB@',
    '@HHHHHH     BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHH       BBBBB@',
    '@RRRRR     P BBBB@',
    '@HHHHH      BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHH P     BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHHH      BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHH       BBBBB@',
    '@@@@@@@@@@@@@@@@@@',]

# n=10
CLEANUP_N10_MAP = [
    '@@@@@@@@@@@@@@@@@@',
    '@RRRRRR     BBBBB@',
    '@HHHHHH    P BBBB@',
    '@RRRRRR     BBBBB@',
    '@RRRRR       BBBB@',
    '@RRRRR      BBBBB@',
    '@HHHH P      BBBB@',
    '@RRRRR      BBBBB@',
    '@HHHHHHSSSSSSBBBB@',
    '@HHHHHHSSSSSSBBBB@',
    '@RRRRR       BBBB@',
    '@HHHHH      BBBBB@',
    '@RRRRRR    P BBBB@',
    '@HHHHHH     BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHH       BBBBB@',
    '@RRRRR     P BBBB@',
    '@HHHHH      BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHH P     BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHHH      BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHH       BBBBB@',
    '@RRRRRR     BBBBB@',
    '@HHHHHH    P BBBB@',
    '@RRRRRR     BBBBB@',
    '@RRRRR       BBBB@',
    '@RRRRR      BBBBB@',
    '@HHHH P      BBBB@',
    '@RRRRR      BBBBB@',
    '@HHHHHHSSSSSSBBBB@',
    '@HHHHHHSSSSSSBBBB@',
    '@RRRRR       BBBB@',
    '@HHHHH      BBBBB@',
    '@RRRRRR    P BBBB@',
    '@HHHHHH     BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHH       BBBBB@',
    '@RRRRR     P BBBB@',
    '@HHHHH      BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHH P     BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHHH      BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHH       BBBBB@',
    '@@@@@@@@@@@@@@@@@@',
]


# n=10
HARVEST_N10_MAP = [
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@ P   P           P          P    P  @',
    '@        A   AA         AAA    A     @',
    '@     A AAA  AAA    A    A AA AAAA   @',
    '@    AAA A    A  A AAA  A  A   A A   @',
    '@    A A       AAA A  AAA            @',
    '@      AAA  AAA  A      AAA   AAA    @',
    '@   P      P          P      P   P   @',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',]

