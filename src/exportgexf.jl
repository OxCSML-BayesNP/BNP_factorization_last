function exportgexf(Z, filename)

  i_idx,j_idx = findnz(Z)
  n = first(size(Z))
  open(filename, "w") do fid
          write(fid, """<?xml version="1.0" encoding="UTF-8"?>""");
          write(fid,""" <gexf xmlns="http://www.gexf.net/1.2draft" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.gexf.net/1.2draft http://www.gexf.net/1.2draft/gexf.xsd" version="1.2">\n """);
          write(fid, """<graph mode="dynamic" defaultedgetype="directed" timeformat="double">\n\n""");
          write(fid, """# Definition of nodes\n""");
          write(fid, """# --------------------\n""");
          write(fid, """<nodes>\n""");
          for k in 1:n
            write(fid, """<node id="$k" />\n""");
          end
          write(fid, """</nodes>\n\n""");
          write(fid, """# Definition of edges\n""");
          write(fid, """# --------------------\n""");
          write(fid, """<edges>\n""");
          for t=1:length(i_idx)
            i = i_idx[t]
            j = j_idx[t]
            write(fid, """<edge source="$i" target="$j">\n""");
            write(fid, """</edge>\n\n""");
          end
          write(fid, """</edges>\n""");
          write(fid, """</graph>\n""");
          write(fid, """</gexf>\n""");
       end
end
