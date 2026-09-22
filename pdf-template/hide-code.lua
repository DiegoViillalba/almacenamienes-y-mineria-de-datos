-- pdf-template/hide-code.lua
-- Filtro Lua de Pandoc para suprimir bloques de código y salidas de terminal en la versión PDF.
-- Convierte el PDF en un texto académico y conceptual, remitiendo al alumno a la web
-- para la ejecución interactiva.

local function is_pdf_target()
    if quarto and quarto.doc then
        return quarto.doc.is_format("pdf") or quarto.doc.is_format("latex")
    end
    if FORMAT then
        return FORMAT:match("latex") or FORMAT:match("pdf")
    end
    return false
end

local function has_class(el, target_class)
    if not el or not el.classes then return false end
    for _, c in ipairs(el.classes) do
        if c == target_class then
            return true
        end
    end
    return false
end

local function is_bibtex(cb)
    if not cb then return false end
    if cb.classes then
        for _, cls in ipairs(cb.classes) do
            local lcls = cls:lower()
            if lcls == "bibtex" or lcls == "bib" then
                return true
            end
        end
    end
    if cb.text and cb.text:match("^%s*@%w+%s*{") then
        return true
    end
    return false
end

local code_languages = {
    python = true,
    py = true,
    sql = true,
    bash = true,
    sh = true,
    shell = true,
    graphql = true,
    yaml = true,
    yml = true,
    r = true,
    scala = true,
    javascript = true,
    js = true,
    typescript = true,
    ts = true,
}

local function is_code_language(cb)
    if not cb or not cb.classes then return false end
    for _, cls in ipairs(cb.classes) do
        if code_languages[cls:lower()] then
            return true
        end
    end
    return false
end

local function filter_codeblock(cb)
    if not is_pdf_target() then
        return nil
    end

    -- Conservar citas BibTeX (como la del capítulo de cómo citar)
    if is_bibtex(cb) then
        return nil
    end

    -- Suprimir mensajes de error de renderizado interactivo (ej. Plotly en PDF)
    if cb.text and cb.text:find("Unable to display output for mime type") then
        return {}
    end

    -- Suprimir bloques de código de celdas ejecutables
    if has_class(cb, "cell-code") or has_class(cb, "sourceCode") then
        return {}
    end

    -- Suprimir bloques de código con lenguajes de programación
    if is_code_language(cb) then
        return {}
    end

    return nil
end

local function has_content(div)
    local found = false
    pandoc.walk_block(div, {
        Image = function(_) found = true end,
        Figure = function(_) found = true end,
        Table = function(_) found = true end,
        RawBlock = function(_) found = true end,
        Str = function(s)
            if s.text:match("%S") then
                found = true
            end
        end,
    })
    return found
end

return {
    {
        -- Pase 1: Suprimir bloques de código, errores de renderizado y salidas de consola
        CodeBlock = filter_codeblock,
        Div = function(div)
            if not is_pdf_target() then return nil end
            if has_class(div, "cell-output-stdout") or has_class(div, "cell-output-stderr") then
                return {}
            end
            return nil
        end,
        Image = function(img)
            if not is_pdf_target() then return nil end
            -- Eliminar atributos explícitos (generados típicamente por Mermaid)
            -- para que \pandocbounded pueda aplicar el límite de 0.7\linewidth
            img.attributes["width"] = nil
            img.attributes["height"] = nil
            return img
        end
    },
    {
        -- Pase 2: Eliminar contenedores de celdas que hayan quedado vacíos tras el Pase 1
        Div = function(div)
            if not is_pdf_target() then return nil end
            if has_class(div, "cell-output-display") or has_class(div, "cell-output") or has_class(div, "cell") then
                if not has_content(div) then
                    return {}
                end
            end
            return nil
        end
    }
}
