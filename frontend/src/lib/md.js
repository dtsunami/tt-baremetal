// Minimal markdown -> HTML for the Kernel Lab docs pane. Input is either our own
// curated refs or live tt-isa-documentation pages (both trusted), so this favours
// small + predictable over exhaustive. Handles: fenced code, headings, pipe tables,
// ordered/unordered lists, blockquotes, hr, bold, inline code, links, images.
//
// Optional `ctx = { rawBase, repoDir }` resolves RELATIVE links/images against the
// ISA repo: images -> rawBase + path (load straight from GitHub); relative *.md
// links -> data-isa="<repo path>" for in-app navigation; other relative files ->
// rawBase absolute. Absolute URLs and #anchors pass through untouched.

function esc(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

// resolve a relative url against ctx.repoDir -> repo-relative path; null if not relative
function resolveRel(url, ctx) {
  if (!ctx || /^[a-z][a-z0-9+.-]*:/i.test(url) || url.startsWith('/') ||
      url.startsWith('#') || url.startsWith('//')) return null
  const stack = ctx.repoDir ? ctx.repoDir.split('/') : []
  for (const part of url.split('/')) {
    if (part === '..') stack.pop()
    else if (part && part !== '.') stack.push(part)
  }
  return stack.join('/')
}

function inline(s, ctx) {
  return s
    .replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (_, a, u) => {
      const rp = resolveRel(u, ctx)
      return `<img alt="${a}" src="${rp ? ctx.rawBase + rp : u}">`
    })
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, t, u) => {
      const rp = resolveRel(u, ctx)
      if (rp) {
        const path = rp.split('#')[0]
        if (path.endsWith('.md')) return `<a href="#isa" data-isa="${path}">${t}</a>`
        return `<a href="${ctx.rawBase + rp}" target="_blank" rel="noopener">${t}</a>`
      }
      return `<a href="${u}" target="_blank" rel="noopener">${t}</a>`
    })
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
}

const BLOCK = /^(#{1,4}\s|```|>\s?|\s*[-*]\s+|\s*\d+\.\s+|---+\s*$)/

export function mdToHtml(md, ctx) {
  const lines = (md || '').replace(/\r\n/g, '\n').split('\n')
  let html = ''
  let i = 0
  while (i < lines.length) {
    const line = lines[i]

    if (/^```/.test(line)) {                                  // fenced code
      const code = []
      i++
      while (i < lines.length && !/^```/.test(lines[i])) code.push(lines[i++])
      i++
      html += `<pre class="code"><code>${esc(code.join('\n'))}</code></pre>`
      continue
    }

    const h = /^(#{1,4})\s+(.*)$/.exec(line)                  // heading
    if (h) { const n = h[1].length; html += `<h${n}>${inline(esc(h[2]), ctx)}</h${n}>`; i++; continue }

    if (/^---+\s*$/.test(line)) { html += '<hr>'; i++; continue }

    // pipe table: header row + |---| separator
    if (line.includes('|') && i + 1 < lines.length &&
        /-/.test(lines[i + 1]) && /^[\s:|-]+$/.test(lines[i + 1])) {
      const row = (r) => r.replace(/^\s*\|/, '').replace(/\|\s*$/, '').split('|').map((c) => c.trim())
      const head = row(line)
      i += 2
      let body = ''
      while (i < lines.length && lines[i].includes('|') && lines[i].trim()) {
        body += '<tr>' + row(lines[i]).map((c) => `<td>${inline(esc(c), ctx)}</td>`).join('') + '</tr>'
        i++
      }
      html += `<table class="dt"><thead><tr>${head.map((c) => `<th>${inline(esc(c), ctx)}</th>`).join('')}</tr></thead><tbody>${body}</tbody></table>`
      continue
    }

    if (/^>\s?/.test(line)) {                                 // blockquote
      const q = []
      while (i < lines.length && /^>\s?/.test(lines[i])) q.push(lines[i++].replace(/^>\s?/, ''))
      html += `<blockquote>${inline(esc(q.join(' ')), ctx)}</blockquote>`
      continue
    }

    if (/^\s*[-*]\s+/.test(line)) {                           // unordered list
      let items = ''
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i]))
        items += `<li>${inline(esc(lines[i++].replace(/^\s*[-*]\s+/, '')), ctx)}</li>`
      html += `<ul>${items}</ul>`
      continue
    }

    if (/^\s*\d+\.\s+/.test(line)) {                          // ordered list
      let items = ''
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i]))
        items += `<li>${inline(esc(lines[i++].replace(/^\s*\d+\.\s+/, '')), ctx)}</li>`
      html += `<ol>${items}</ol>`
      continue
    }

    if (!line.trim()) { i++; continue }                       // blank

    const buf = []                                            // paragraph
    while (i < lines.length && lines[i].trim() && !BLOCK.test(lines[i])) buf.push(lines[i++])
    html += `<p>${inline(esc(buf.join(' ')), ctx)}</p>`
  }
  return html
}
