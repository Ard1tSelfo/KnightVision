def chess_notation_to_index(coordinate):
    # Maps 'a' to 0, 'b' to 1, ..., 'h' to 7
    col_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
    # Extract column letter and row number from the coordinate
    col, row = coordinate[0].lower(), int(coordinate[1])
    # Calculate index (note: rows are from 8 at top to 1 at bottom, hence "8-row")
    index = (8 - row) * 8 + col_map[col]
    return index

def build_fen(predicted_symbols, coordinates, active_turn):

    indices = [chess_notation_to_index(coord) for coord in coordinates]
    # Combine the indices and predicted_symbols for sorting
    combined = list(zip(indices, predicted_symbols))
    # Sort by chessboard index
    sorted_combined = sorted(combined, key=lambda x: x[0])

    # Extract the sorted predicted symbols
    sorted_predicted_symbols = [symbol for _, symbol in sorted_combined]
    fen_rows = []
    castling_rights = {'K': False, 'Q': False, 'k': False, 'q': False}  # Initialize castling rights

    # Check for castling availability based on static positions
    if sorted_predicted_symbols[60] == 'K' and sorted_predicted_symbols[63] == 'R':
        castling_rights['K'] = True
    if sorted_predicted_symbols[60] == 'K' and sorted_predicted_symbols[56] == 'R':
        castling_rights['Q'] = True
    if sorted_predicted_symbols[4] == 'k' and sorted_predicted_symbols[7] == 'R':
        castling_rights['k'] = True
    if sorted_predicted_symbols[4] == 'K' and sorted_predicted_symbols[0] == 'R':
        castling_rights['q'] = True

    for row in range(8):
        empty_count = 0
        fen_row = ''
        for col in range(8):
            symbol = sorted_predicted_symbols[row * 8 + col]
            if symbol == '1':  # Empty square
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += symbol
        if empty_count > 0:  # End of row empty squares
            fen_row += str(empty_count)
        fen_rows.append(fen_row)

    castling_availability = ''.join([k for k, v in castling_rights.items() if v])
    if not castling_availability:
        castling_availability = '-'

    # Add active turn ('w' for White, 'b' for Black) at the end
    active_color = 'w' if active_turn else 'b'

    fen = '/'.join(fen_rows) + " " + active_color + " " + castling_availability + " -"

    return fen
