-- pdf-template/consolidate-refs.lua
-- Filtro Lua de Pandoc para consolidar la bibliografía en una sola sección final al compilar a PDF.
-- Elimina las secciones intermedias "## Referencias" y "::: {#refs}" dentro de los capítulos individuales,
-- asegurando que la bibliografía completa solo se genere una vez al final del libro.

function Pandoc(doc)
    -- Solo aplicar cuando se compila a PDF o LaTeX
    if quarto and quarto.doc and not (quarto.doc.is_format("pdf") or quarto.doc.is_format("latex")) then
        return doc
    end

    -- Contar el total de bloques con id="refs"
    local total_refs = 0
    for _, el in ipairs(doc.blocks) do
        if el.t == "Div" and el.identifier == "refs" then
            total_refs = total_refs + 1
        end
    end

    -- Si hay 1 o menos bloques de referencias, no hay nada que consolidar
    if total_refs <= 1 then
        return doc
    end

    local new_blocks = {}
    local refs_seen = 0
    local i = 1
    local n = #doc.blocks

    while i <= n do
        local el = doc.blocks[i]

        -- Caso 1: Regla horizontal inmediatamente anterior al encabezado de Referencias
        if el.t == "HorizontalRule" and i + 1 <= n and doc.blocks[i+1].t == "Header" then
            local next_h = doc.blocks[i+1]
            local h_text = pandoc.utils.stringify(next_h):lower():gsub("^%s*(.-)%s*$", "%1")
            if (h_text == "referencias" or h_text == "references") and i + 2 <= n then
                local next_div = doc.blocks[i+2]
                if next_div.t == "Div" and next_div.identifier == "refs" then
                    if refs_seen + 1 < total_refs then
                        refs_seen = refs_seen + 1
                        i = i + 3
                        goto continue
                    end
                end
            end
        end

        -- Caso 2: Encabezado de Referencias seguido directamente del bloque refs
        if el.t == "Header" then
            local h_text = pandoc.utils.stringify(el):lower():gsub("^%s*(.-)%s*$", "%1")
            if (h_text == "referencias" or h_text == "references") and i + 1 <= n then
                local next_div = doc.blocks[i+1]
                if next_div.t == "Div" and next_div.identifier == "refs" then
                    if refs_seen + 1 < total_refs then
                        refs_seen = refs_seen + 1
                        i = i + 2
                        goto continue
                    end
                end
            end
        end

        -- Caso 3: Bloque refs huérfano intermedio (no el último)
        if el.t == "Div" and el.identifier == "refs" then
            if refs_seen + 1 < total_refs then
                refs_seen = refs_seen + 1
                i = i + 1
                goto continue
            end
        end

        table.insert(new_blocks, el)
        i = i + 1
        ::continue::
    end

    doc.blocks = new_blocks
    return doc
end
